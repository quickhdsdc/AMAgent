import base64
import urllib.parse
from app.aas_utils.basyx_client import BasyxApiClient
import os
from pyecma376_2 import ZipPackageReader
from basyx.aas import model
from basyx.aas.adapter import aasx
from basyx.aas.adapter.xml import read_aas_xml_file
from basyx.aas.adapter.json import read_aas_json_file
import io
import json
import logging
import re
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
import networkx as nx
from dateutil import parser as date_parser


def get_external_related_parts(aasx_filepath, rel_type_filter=None):
    rels_path = "aasx/_rels/aasx-origin.rels"

    with zipfile.ZipFile(aasx_filepath, "r") as zipf:
        if rels_path not in zipf.namelist():
            return []

        with zipf.open(rels_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

            ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
            matches = []

            for rel in root.findall("rel:Relationship", ns):
                rel_type = rel.attrib.get("Type")
                target = rel.attrib.get("Target")
                target_mode = rel.attrib.get("TargetMode", "Internal")

                if rel_type_filter is None or rel_type == rel_type_filter:
                    matches.append({
                        "type": rel_type,
                        "target": target,
                        "mode": target_mode
                    })

            return matches


def encode_id(id_str: str) -> str:
    b64_raw = base64.urlsafe_b64encode(id_str.encode()).decode()
    return urllib.parse.quote(b64_raw, safe='')


async def _enumerate_submodel_ids(client: BasyxApiClient, b64_aas_id: str):
    """Return the list of submodel ids referenced by a shell, or [] on failure.
    BaSyx's /serialization endpoint returns a shell-only package when the
    submodelIds query param is empty, so we always enumerate first."""
    try:
        refs = await client.get(f"/shells/{b64_aas_id}/submodel-refs")
        return [r["keys"][0]["value"] for r in (refs or {}).get("result", [])]
    except Exception:
        return []


async def get_aasx(endpoint, aas_id, base_dir):

    client = BasyxApiClient(endpoint)
    b64_aas_id = encode_id(aas_id)

    shell_metadata = await client.get(f"/shells/{b64_aas_id}")
    id_short = (shell_metadata or {}).get("idShort")
    if not id_short:
        id_short = aas_id.split("/")[-1]

    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, id_short + ".aasx")

    submodel_ids = await _enumerate_submodel_ids(client, b64_aas_id)
    try:
        result = await client.download_aas_package(aas_id, submodel_ids, filepath)
    except Exception:
        result = None
    if not result or not os.path.exists(filepath):
        result = await client.download_aas_package(aas_id, [], filepath)
    if not result or not os.path.exists(filepath):
        raise RuntimeError(
            f"AASX download did not produce local file at '{filepath}' for aas_id '{aas_id}'"
        )
    return filepath


async def get_json(endpoint, aas_id, base_dir):

    client = BasyxApiClient(endpoint)
    b64_aas_id = encode_id(aas_id)

    shell_metadata = await client.get(f"/shells/{b64_aas_id}")
    id_short = (shell_metadata or {}).get("idShort")
    if not id_short:
        id_short = aas_id.split("/")[-1]

    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, id_short + ".json")

    submodel_ids = await _enumerate_submodel_ids(client, b64_aas_id)
    result = await client.download_aas_json(aas_id, submodel_ids, filepath)
    if not result or not os.path.exists(filepath):
        raise RuntimeError(
            f"JSON download did not produce local file at '{filepath}' for aas_id '{aas_id}'"
        )
    return filepath


def _write_graph_json(G: nx.DiGraph, graph_path: str) -> None:
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=2)


_GLOBALLY_IDENTIFIABLE_KEY_TYPES = frozenset({"GlobalReference", "FragmentReference"})


def _sanitize_aas_structural(data):
    """Structural fixes for BaSyx Java AAS env JSON that basyx-python-sdk 2.0.0's
    strict deserializer rejects. Covers wrong-type-on-whole-slot cases that the
    leaf value pass can't see: AAS.submodel[*].type null/ExternalReference (should
    be ModelReference), AAS.assetInformation null (AASd-131), and ExternalReference
    keys whose first entry has a non-globally-identifiable type (AASd-122)."""
    if not isinstance(data, dict):
        return data

    for shell in data.get("assetAdministrationShells", []) or []:
        if not isinstance(shell, dict):
            continue
        for ref in shell.get("submodels", []) or []:
            if isinstance(ref, dict) and ref.get("type") in (None, "ExternalReference"):
                ref["type"] = "ModelReference"
        ai = shell.get("assetInformation")
        if ai is None:
            shell["assetInformation"] = {
                "assetKind": "Instance",
                "globalAssetId": shell.get("id") or "urn:placeholder",
            }
        elif isinstance(ai, dict):
            if not ai.get("globalAssetId") and not ai.get("specificAssetIds"):
                ai["globalAssetId"] = shell.get("id") or "urn:placeholder"

    def _walk_aasd122(o):
        if isinstance(o, dict):
            if o.get("type") == "ExternalReference":
                keys = o.get("keys")
                if isinstance(keys, list) and keys and isinstance(keys[0], dict):
                    if keys[0].get("type") not in _GLOBALLY_IDENTIFIABLE_KEY_TYPES:
                        keys[0]["type"] = "GlobalReference"
            for v in o.values():
                _walk_aasd122(v)
        elif isinstance(o, list):
            for x in o:
                _walk_aasd122(x)
    _walk_aasd122(data)
    return data


def sanitize_aas_json_values(obj):
    """Recursively clean invalid Property values, idShorts, MultiLanguageText entries,
    and empty strings/references so basyx-python-sdk 2.0.0's strict validators accept
    the document. Mutates in place and returns the same object for chaining."""
    if isinstance(obj, dict):
        val_type = obj.get("valueType") or obj.get("value_type")
        val = obj.get("value")

        if val_type and val is not None:
            val_str = str(val).strip()
            if val_type in ("xs:boolean", "boolean"):
                if val_str.lower() not in ("true", "false", "1", "0"):
                    obj["valueType"] = "xs:string"
            elif val_type in (
                "xs:int", "xs:integer", "xs:long", "xs:short", "xs:byte",
                "xs:unsignedInt", "xs:unsignedLong", "xs:unsignedShort", "xs:unsignedByte",
                "xs:nonNegativeInteger", "xs:positiveInteger",
                "xs:nonPositiveInteger", "xs:negativeInteger",
                "int", "integer",
            ):
                try:
                    int(float(val_str))
                except ValueError:
                    obj["valueType"] = "xs:string"
            elif val_type in ("xs:float", "xs:double", "xs:decimal", "float", "double"):
                try:
                    float(val_str)
                except ValueError:
                    obj["valueType"] = "xs:string"
            elif val_type in (
                "xs:date", "xs:dateTime", "xs:time",
                "xs:gYear", "xs:gYearMonth", "xs:gMonth", "xs:gMonthDay", "xs:gDay",
                "date", "dateTime", "time",
                "gYear", "gYearMonth", "gMonth", "gMonthDay", "gDay",
            ):
                if val_str:
                    try:
                        date_parser.parse(val_str)
                    except Exception:
                        obj["valueType"] = "xs:string"
                else:
                    obj["value"] = None
                    obj["valueType"] = "xs:string"

        if "idShort" in obj and obj["idShort"]:
            new_id_short = re.sub(r"[^a-zA-Z0-9_]", "_", obj["idShort"])
            if new_id_short != obj["idShort"]:
                obj["idShort"] = new_id_short

        if "shortName" in obj and isinstance(obj["shortName"], list):
            for item in obj["shortName"]:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    if text is None:
                        item["text"] = "?"
                    elif len(text) > 18:
                        item["text"] = text[:18]
                    elif len(text) == 0:
                        item["text"] = "?"

        for v in obj.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "language" in item and "text" in item:
                        lang = re.sub(r"[^a-zA-Z]", "", str(item.get("language", "")))
                        item["language"] = lang[:2].lower() if len(lang) >= 2 else "en"
                        text = item.get("text")
                        if text is None or len(text) == 0:
                            item["text"] = "?"

        for key in ("dataType", "valueFormat", "valueType"):
            if key in obj and isinstance(obj[key], str) and len(obj[key]) == 0:
                obj[key] = "STRING"

        keys_to_delete = []
        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, str) and len(v) == 0:
                obj[k] = "_"
            elif isinstance(v, dict) and "keys" in v and isinstance(v["keys"], list) and len(v["keys"]) == 0:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del obj[k]

        for v in obj.values():
            sanitize_aas_json_values(v)

    elif isinstance(obj, list):
        for item in obj:
            sanitize_aas_json_values(item)
    return obj


def sanitize_aas_xml_values(xml_content: bytes) -> bytes:
    """Sanitize values in AAS XML so basyx-python-sdk 2.0.0's strict parser
    accepts it: invalid numeric/boolean/date Property values fall back to
    xs:string, idShorts are normalised, shortName entries clamped to 1..18 chars.
    Returns the original bytes on any parse failure (best-effort)."""
    try:
        events = io.BytesIO(xml_content)
        for _, (prefix, uri) in ET.iterparse(events, events=["start-ns"]):
            ET.register_namespace(prefix, uri)
    except Exception:
        pass
    ET.register_namespace("aas", "https://admin-shell.io/aas/3/0")
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    try:
        root = ET.fromstring(xml_content)
        for elem in root.iter():
            val_type_elem = None
            value_elem = None
            for child in elem:
                tag = child.tag
                if tag.endswith("valueType"):
                    val_type_elem = child
                elif tag.endswith("value"):
                    value_elem = child

            if val_type_elem is not None and value_elem is not None:
                val_type = val_type_elem.text
                val_text = value_elem.text
                is_invalid = False
                if val_type and val_text is not None:
                    s_val = val_text.strip()
                    if val_type in ("xs:boolean", "boolean"):
                        if s_val.lower() not in ("true", "false", "1", "0"):
                            is_invalid = True
                    elif val_type in (
                        "xs:int", "xs:integer", "xs:long", "xs:short", "xs:byte",
                        "xs:unsignedInt", "xs:unsignedLong", "xs:unsignedShort", "xs:unsignedByte",
                        "xs:nonNegativeInteger", "xs:positiveInteger",
                        "xs:nonPositiveInteger", "xs:negativeInteger",
                        "int", "integer",
                    ):
                        try:
                            if s_val == "":
                                is_invalid = True
                            else:
                                int(float(s_val))
                        except ValueError:
                            is_invalid = True
                    elif val_type in ("xs:float", "xs:double", "xs:decimal", "float", "double"):
                        try:
                            if s_val == "":
                                is_invalid = True
                            else:
                                float(s_val)
                        except ValueError:
                            is_invalid = True
                elif val_type and val_text is None:
                    if val_type in (
                        "xs:boolean", "boolean",
                        "xs:int", "xs:integer", "int", "integer",
                        "xs:float", "xs:double", "float", "double",
                        "xs:date", "xs:dateTime", "xs:time",
                        "xs:gYear", "xs:gYearMonth", "xs:gMonth", "xs:gMonthDay", "xs:gDay",
                        "date", "dateTime", "time",
                        "gYear", "gYearMonth", "gMonth", "gMonthDay", "gDay",
                    ):
                        is_invalid = True
                if not is_invalid and val_text is not None and val_type in (
                    "xs:date", "xs:dateTime", "xs:time",
                    "xs:gYear", "xs:gYearMonth", "xs:gMonth", "xs:gMonthDay", "xs:gDay",
                    "date", "dateTime", "time",
                    "gYear", "gYearMonth", "gMonth", "gMonthDay", "gDay",
                ):
                    try:
                        date_parser.parse(val_text)
                    except Exception:
                        is_invalid = True
                if is_invalid:
                    val_type_elem.text = "xs:string"

            if elem.tag.endswith("shortName"):
                for lang_string in elem:
                    for child in lang_string:
                        if child.tag.endswith("text"):
                            if child.text is None or len(child.text) == 0:
                                child.text = "?"
                            elif len(child.text) > 18:
                                child.text = child.text[:18]
                            break

            for child in elem:
                if child.tag.endswith("idShort") and child.text:
                    new_id_short = re.sub(r"[^a-zA-Z0-9_]", "_", child.text)
                    if new_id_short != child.text:
                        child.text = new_id_short

        out = io.BytesIO()
        ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
        return out.getvalue()
    except Exception as e:
        logging.getLogger(__name__).warning(f"XML sanitization fell back to raw bytes: {e}")
        return xml_content


def aasx_parser(aasx_filepath: str) -> pd.DataFrame:
    """Parse a downloaded AASX package into the flat DataFrame and emit a
    parallel NetworkX DiGraph (saved alongside as `<stem>_graph.json`).
    The graph preserves explicit parent-child containment edges across the
    AAS / Submodel / SMC / SubmodelElement hierarchy, which the upstream
    retrieval tool uses for structural context expansion."""
    aas_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore()
    file_store = aasx.DictSupplementaryFileContainer()
    with ZipPackageReader(aasx_filepath) as reader:
        rel_type = "http://admin-shell.io/aasx/relationships/aas-spec"
        related_parts = get_external_related_parts(aasx_filepath, rel_type_filter=rel_type)
        if related_parts:
            for rel in related_parts:
                aas_part = rel.get("target")
                if not aas_part:
                    continue
                if aas_part.endswith(".xml"):
                    with reader.open_part(aas_part) as p:
                        sanitized = sanitize_aas_xml_values(p.read())
                        f_pseudo = io.BytesIO(sanitized)
                        logging.getLogger("basyx").setLevel(logging.CRITICAL)
                        try:
                            aas_objs = read_aas_xml_file(f_pseudo, failsafe=True)
                        finally:
                            logging.getLogger("basyx").setLevel(logging.WARNING)
                        for obj in aas_objs:
                            aas_store.add(obj)
                elif aas_part.endswith(".json"):
                    with reader.open_part(aas_part) as p:
                        raw = io.TextIOWrapper(p, encoding="utf-8-sig").read()
                        data = json.loads(raw)
                        data = _sanitize_aas_structural(data)
                        data = sanitize_aas_json_values(data)
                        f_pseudo = io.StringIO(json.dumps(data))
                        logging.getLogger("basyx").setLevel(logging.CRITICAL)
                        try:
                            aas_objs = read_aas_json_file(f_pseudo, failsafe=True)
                        finally:
                            logging.getLogger("basyx").setLevel(logging.WARNING)
                        for obj in aas_objs:
                            aas_store.add(obj)
                else:
                    print(f"Unsupported file format: {aas_part}")

    df_aas, G = flatten_aas_object_store(aas_store, return_graph=True)
    csv_path = aasx_filepath.replace(".aasx", ".csv")
    df_aas.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_graph_json(G, aasx_filepath.replace(".aasx", "_graph.json"))
    return df_aas


def aas_json_parser(aasx_filepath: str) -> pd.DataFrame:
    aas_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore()
    with open(aasx_filepath, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    data = _sanitize_aas_structural(data)
    data = sanitize_aas_json_values(data)
    f_pseudo = io.StringIO(json.dumps(data))
    logging.getLogger("basyx").setLevel(logging.CRITICAL)
    try:
        aas_objs = read_aas_json_file(f_pseudo, failsafe=True)
    finally:
        logging.getLogger("basyx").setLevel(logging.WARNING)
    for obj in aas_objs:
        aas_store.add(obj)
    df_aas, G = flatten_aas_object_store(aas_store, return_graph=True)
    csv_path = aasx_filepath.replace(".json", ".csv")
    df_aas.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_graph_json(G, aasx_filepath.replace(".json", "_graph.json"))
    return df_aas


def flatten_aas_object_store(
    object_store: model.DictObjectStore[model.Identifiable],
    with_entity: bool = False,
    return_graph: bool = False,
):
    """Flatten an AAS object store into (a) a DataFrame of entities (legacy
    flat-list view) and (b) a NetworkX DiGraph where edges encode the
    'has-component' containment relation (AAS → Submodel → SMC → SME).
    ConceptDescriptions are resolved per entity and attached as
    `cd_definition` / `cd_unit` attributes on the graph nodes."""

    def get_description_text(desc_dict, language="en"):
        if not desc_dict:
            return "None"
        try:
            if language in desc_dict:
                return desc_dict[language]
            return next(iter(desc_dict.values()), "None")
        except Exception:
            return "None"

    def get_semantic_id_str(sem_id):
        if sem_id is None:
            return ""
        try:
            keys = getattr(sem_id, "key", None)
            if keys and len(keys) > 0:
                return str(keys[0].value)
        except Exception:
            pass
        return ""

    cds = {}
    for obj in object_store:
        if isinstance(obj, model.ConceptDescription):
            cd_id = obj.id
            desc = get_description_text(getattr(obj, "description", None))
            unit = ""
            for eds in getattr(obj, "embedded_data_specifications", []):
                ds_content = getattr(eds, "data_specification_content", None)
                if ds_content and hasattr(ds_content, "unit"):
                    unit_val = getattr(ds_content, "unit", "")
                    if unit_val:
                        unit = str(unit_val)
            cds[cd_id] = {"definition": desc, "unit": unit}

    G = nx.DiGraph()
    rows = []

    for identifiable in object_store:
        if not isinstance(identifiable, model.AssetAdministrationShell):
            continue

        aas = identifiable
        aas_id = aas.id
        aas_id_short = aas.id_short
        encoded_aas_id = encode_id(aas_id)

        aas_node = aas_id_short
        G.add_node(
            aas_node,
            idShort=aas_id_short,
            type=type(aas).__name__,
            description=get_description_text(aas.description),
            value=None,
            API_path=f"/shells/{encoded_aas_id}",
            semanticId="None",
            cd_definition="",
            cd_unit="",
        )
        rows.append({
            "idShort": aas_id_short,
            "type": type(aas).__name__,
            "description": get_description_text(aas.description),
            "value": None,
            "semantic_path": f"{aas_id_short}",
            "API_path": f"/shells/{encoded_aas_id}",
            "semanticId": "None",
            "cd_definition": "",
            "cd_unit": "",
        })

        for ref in aas.submodel:
            submodel_id = ref.key[0].value
            submodel = object_store.get(submodel_id)
            if not isinstance(submodel, model.Submodel):
                continue

            sub_id_short = submodel.id_short
            sub_descrip = get_description_text(submodel.description)
            semantic_path_sm = f"{aas_id_short}/{sub_id_short}"
            sub_sem = get_semantic_id_str(submodel.semantic_id)
            sub_cd = cds.get(sub_sem, {})

            sub_node = f"{aas_id_short}/{sub_id_short}"
            G.add_node(
                sub_node,
                idShort=sub_id_short,
                type=type(submodel).__name__,
                description=sub_descrip,
                value=None,
                API_path=f"/submodels/{encode_id(submodel.id)}",
                semanticId=sub_sem,
                cd_definition=sub_cd.get("definition", ""),
                cd_unit=sub_cd.get("unit", ""),
            )
            G.add_edge(aas_node, sub_node)
            rows.append({
                "idShort": sub_id_short,
                "type": type(submodel).__name__,
                "description": sub_descrip,
                "value": None,
                "semantic_path": semantic_path_sm,
                "API_path": f"/submodels/{encode_id(submodel.id)}",
                "semanticId": sub_sem,
                "cd_definition": sub_cd.get("definition", ""),
                "cd_unit": sub_cd.get("unit", ""),
            })

            def _flatten_element(elem: model.SubmodelElement, parent_path: str, parent_node_id: str, visited=None):
                if visited is None:
                    visited = set()
                obj_id = id(elem)
                if obj_id in visited:
                    return
                visited.add(obj_id)

                elem_id_short = elem.id_short
                elem_type = type(elem).__name__
                elem_desc = get_description_text(getattr(elem, "description", []))
                elem_sem = get_semantic_id_str(getattr(elem, "semantic_id", None))
                elem_cd = cds.get(elem_sem, {})

                elem_val = None
                if isinstance(elem, model.Property):
                    elem_val = str(elem.value) if elem.value is not None else None
                elif isinstance(elem, model.MultiLanguageProperty):
                    try:
                        val = getattr(elem, "value", None)
                        if val:
                            texts = (
                                [f"{ls.language}: {ls.text}" for ls in val]
                                if hasattr(val, "__iter__")
                                else str(val)
                            )
                            elem_val = str(texts)
                    except Exception:
                        elem_val = str(getattr(elem, "value", ""))
                elif isinstance(elem, model.File):
                    elem_val = str(elem.value) if elem.value is not None else None
                elif isinstance(elem, model.Range):
                    elem_val = f"min={elem.min}, max={elem.max}"
                elif isinstance(elem, model.ReferenceElement):
                    ref_val = elem.value
                    if ref_val:
                        keys = [f"{k.type}={k.value}" for k in ref_val.keys]
                        elem_val = ";".join(keys)

                full_id_path = elem_id_short if parent_path == "" else f"{parent_path}.{elem_id_short}"
                elem_path = f"/submodels/{encode_id(submodel_id)}/submodel-elements/{full_id_path}"
                elem_node_id = f"{aas_id_short}/{sub_id_short}/{full_id_path}"

                G.add_node(
                    elem_node_id,
                    idShort=elem_id_short,
                    type=elem_type,
                    description=elem_desc,
                    value=elem_val,
                    API_path=elem_path,
                    semanticId=elem_sem,
                    cd_definition=elem_cd.get("definition", ""),
                    cd_unit=elem_cd.get("unit", ""),
                )
                G.add_edge(parent_node_id, elem_node_id)
                rows.append({
                    "idShort": elem_id_short,
                    "type": elem_type,
                    "description": elem_desc,
                    "value": elem_val,
                    "semantic_path": f"{semantic_path_sm}/{full_id_path}",
                    "API_path": elem_path,
                    "semanticId": elem_sem,
                    "cd_definition": elem_cd.get("definition", ""),
                    "cd_unit": elem_cd.get("unit", ""),
                })

                if isinstance(elem, (model.SubmodelElementCollection, model.SubmodelElementList)):
                    for child in getattr(elem, "value", []) or []:
                        _flatten_element(child, full_id_path, elem_node_id, visited)
                elif isinstance(elem, model.Entity) and with_entity:
                    for child in getattr(elem, "statement", []) or []:
                        child_sem = get_semantic_id_str(getattr(child, "semantic_id", None))
                        child_cd = cds.get(child_sem, {})
                        c_node_id = f"{elem_node_id}/{child.id_short}"
                        child_val = (
                            getattr(child, "value", None)
                            if hasattr(child, "value")
                            and not isinstance(child, (model.SubmodelElementCollection, model.SubmodelElementList))
                            else None
                        )
                        if child_val is not None and not isinstance(child_val, (str, int, float, bool, list, dict)):
                            child_val = str(child_val)
                        G.add_node(
                            c_node_id,
                            idShort=child.id_short,
                            type=type(child).__name__,
                            description=get_description_text(getattr(child, "description", [])),
                            value=child_val,
                            API_path=f"{elem_path}.{child.id_short}",
                            semanticId=child_sem,
                            cd_definition=child_cd.get("definition", ""),
                            cd_unit=child_cd.get("unit", ""),
                        )
                        G.add_edge(elem_node_id, c_node_id)
                        rows.append({
                            "idShort": child.id_short,
                            "type": type(child).__name__,
                            "description": get_description_text(getattr(child, "description", [])),
                            "value": child_val,
                            "semantic_path": f"/{submodel_id}/{full_id_path}/{child.id_short}",
                            "API_path": f"{elem_path}.{child.id_short}",
                            "semanticId": child_sem,
                            "cd_definition": child_cd.get("definition", ""),
                            "cd_unit": child_cd.get("unit", ""),
                        })

            for elem in submodel.submodel_element:
                _flatten_element(elem, "", sub_node)

    df = pd.DataFrame(
        rows,
        columns=[
            "idShort",
            "type",
            "description",
            "value",
            "semantic_path",
            "API_path",
            "semanticId",
            "cd_definition",
            "cd_unit",
        ],
    )
    if return_graph:
        return df, G
    return df
