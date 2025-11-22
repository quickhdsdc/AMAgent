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
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET


def get_external_related_parts(aasx_filepath, rel_type_filter=None):
    rels_path = "aasx/_rels/aasx-origin.rels"

    with zipfile.ZipFile(aasx_filepath, "r") as zipf:
        if rels_path not in zipf.namelist():
            return []

        with zipf.open(rels_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

            # Define namespace
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


async def get_aasx(endpoint, aas_id, base_dir):

    client = BasyxApiClient(endpoint)
    b64_aas_id = encode_id(aas_id)
    # Get submodel references
    submodel_refs = await client.get(f"/shells/{b64_aas_id}/submodel-refs")
    submodel_refs = submodel_refs.get("result", [])
    submodel_ids = [ref["keys"][0]["value"] for ref in submodel_refs]

    os.makedirs(base_dir, exist_ok=True)
    filename = aas_id.split("/")[-1] + ".aasx"
    filepath = os.path.join(base_dir, filename)

    # Download the package
    await client.download_aas_package(aas_id, submodel_ids, filepath)
    print(f"Downloaded AASX file to {filepath}")

    return filepath


async def get_json(endpoint, aas_id, base_dir):

    client = BasyxApiClient(endpoint)
    b64_aas_id = encode_id(aas_id)
    # Get submodel references
    submodel_refs = await client.get(f"/shells/{b64_aas_id}/submodel-refs")
    submodel_refs = submodel_refs.get("result", [])
    submodel_ids = [ref["keys"][0]["value"] for ref in submodel_refs]

    os.makedirs(base_dir, exist_ok=True)
    filename = aas_id.split("/")[-1] + ".json"
    filepath = os.path.join(base_dir, filename)

    # Download the package
    await client.download_aas_json(aas_id, submodel_ids, filepath)
    print(f"Downloaded json file to {filepath}")

    return filepath


def aasx_parser(aasx_filepath: str) -> pd.DataFrame:
    aas_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore()
    file_store = aasx.DictSupplementaryFileContainer()
    with ZipPackageReader(aasx_filepath) as reader:
        rel_type = "http://admin-shell.io/aasx/relationships/aas-spec"
        related_parts = get_external_related_parts(aasx_filepath, rel_type_filter=rel_type)
        if related_parts:
            for rel in related_parts:
                aas_part = rel.get("target")
                if not aas_part:
                    continue  # skip invalid
                if aas_part.endswith(".xml"):
                    with reader.open_part(aas_part) as p:
                        aas_objs = read_aas_xml_file(p)
                        for obj in aas_objs:
                            aas_store.add(obj)
                elif aas_part.endswith(".json"):
                    with reader.open_part(aas_part) as p:
                        aas_objs = read_aas_json_file(io.TextIOWrapper(p, encoding='utf-8-sig'))
                        for obj in aas_objs:
                            aas_store.add(obj)
                else:
                    print(f"Unsupported file format: {aas_part}")
    df_aas = flatten_aas_object_store(aas_store)
    csv_path = aasx_filepath.replace(".aasx", ".csv")
    df_aas.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_aas


def aas_json_parser(aasx_filepath: str) -> pd.DataFrame:
    aas_store: model.DictObjectStore[model.Identifiable] = model.DictObjectStore()
    with open(aasx_filepath, "rb") as f:
        aas_objs = read_aas_json_file(io.TextIOWrapper(f, encoding='utf-8-sig'))
    for obj in aas_objs:
        aas_store.add(obj)
    df_aas = flatten_aas_object_store(aas_store)
    csv_path = aasx_filepath.replace(".json", ".csv")
    df_aas.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_aas



def flatten_aas_object_store(object_store: model.DictObjectStore[model.Identifiable], with_entity: bool = False) -> pd.DataFrame:

    def get_description_text(desc_dict, language='en'):
        if not desc_dict:
            return 'None'
        try:
            # Try to get the preferred language
            if language in desc_dict:
                return desc_dict[language]
            # Fallback: return the first available language's text
            return next(iter(desc_dict.values()), 'None')
        except Exception:
            return 'None'

    # Helper function to extract a semanticId string (e.g., first key’s value if available)
    def get_semantic_id_str(sem_id):
        if sem_id is None:
            return ""
        try:
            keys = getattr(sem_id, 'key', None)
            if keys and len(keys) > 0:
                return str(keys[0].value)
        except Exception:
            pass
        return ""

    rows = []

    for identifiable in object_store:
        if isinstance(identifiable, model.AssetAdministrationShell):
            aas = identifiable
            aas_id = aas.id
            aas_id_short = aas.id_short
            encoded_aas_id = encode_id(aas_id)

            rows.append({
                "idShort": aas_id_short,
                "type": type(aas).__name__,
                "description": get_description_text(aas.description),
                "semantic_path": f"{aas_id_short}",
                "API_path": f"/shells/{encoded_aas_id}",
                "semanticId": 'None'
            })

            # For each submodel reference in the AAS
            for ref in aas.submodel:
                submodel_id = ref.key[0].value
                submodel = object_store.get(submodel_id)
                if not isinstance(submodel, model.Submodel):
                    continue

                sub_id_short = submodel.id_short
                sub_descrip = get_description_text(submodel.description)
                semantic_path_sm = f"{aas_id_short}/{sub_id_short}"
                rows.append({
                    "idShort": sub_id_short,
                    "type": type(submodel).__name__,
                    "description": sub_descrip,
                    "semantic_path": semantic_path_sm,
                    "API_path": f"/submodels/{encode_id(submodel.id)}",
                    "semanticId": get_semantic_id_str(submodel.semantic_id)
                })

                def _flatten_element(elem: model.SubmodelElement, parent_path: str):
                    elem_id_short = elem.id_short
                    elem_type = type(elem).__name__
                    elem_desc = get_description_text(getattr(elem, 'description', []))
                    elem_sem = get_semantic_id_str(getattr(elem, 'semantic_id', None))
                    # Build the path segment for this element
                    full_id_path = elem_id_short if parent_path == "" else f"{parent_path}.{elem_id_short}"
                    elem_path = f"/submodels/{encode_id(submodel_id)}/submodel-elements/{full_id_path}"
                    # Add this element's info to the list
                    rows.append({
                        "idShort": elem_id_short,
                        "type": elem_type,
                        "description": elem_desc,
                        "semantic_path": f"{semantic_path_sm}/{full_id_path}",
                        "API_path": elem_path,
                        "semanticId": elem_sem
                    })
                    if isinstance(elem, (model.SubmodelElementCollection, model.SubmodelElementList)):
                        for child in getattr(elem, 'value', []) or []:
                            _flatten_element(child, full_id_path)
                    elif isinstance(elem, model.Entity):
                        if with_entity:
                            for child in getattr(elem, 'statement', []) or []:
                                rows.append({
                                    "idShort": child.id_short,
                                    "type": type(child).__name__,
                                    "description": get_description_text(getattr(child, 'description', [])),
                                    "semantic_path": f"/{submodel_id}/{full_id_path}/{child.id_short}",
                                    "API_path": f"{elem_path}.{child.id_short}",
                                    "semanticId": get_semantic_id_str(getattr(child, 'semantic_id', None))
                                })

                for elem in submodel.submodel_element:
                    _flatten_element(elem, "")

    return pd.DataFrame(rows, columns=["idShort", "type", "description", "semantic_path", "API_path", "semanticId"])