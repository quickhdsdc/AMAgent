from app.tool.base import BaseTool, ToolResult
from app.aas_utils.basyx_client import BasyxApiClient
from app.aas_utils import aas_loader

import os
import re
import json
import time
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from langchain_openai import AzureOpenAIEmbeddings


_CONTAINER_TYPES = {
    "AssetAdministrationShell",
    "Submodel",
    "SubmodelElementCollection",
    "SubmodelElementList",
}

_RESULTS_DIR = Path("./results_AM")


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _as_obj(text: str) -> Any:
    """Parse the first complete JSON value found in `text` (tolerant of fences / noise)."""
    s = _strip_fences(text)
    decoder = json.JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] in "{[":
            try:
                obj, _ = decoder.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
                i += 1
                continue
        i += 1
    raise ValueError("No complete JSON object/array found in input.")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode()


def _save_evidence(evidence: dict, filename: str = "knowledge_build_context.json") -> str:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    return str(out_path)


def _embedder() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDINGS_MODEL", "text-embedding-3-large"),
        azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large-1"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


async def _aas_explore(endpoint: str) -> List[Dict[str, Any]]:
    """Enumerate all shells and their submodel descriptors at `endpoint`."""
    client = BasyxApiClient(endpoint)
    shells = await client.get_shells()
    if not isinstance(shells, list):
        return []
    out = []
    for sh in shells:
        aas_id = sh.get("id")
        if not aas_id:
            continue
        b64 = base64.urlsafe_b64encode(aas_id.encode()).decode()
        refs = await client.get(f"/shells/{b64}/submodel-refs")
        refs = (refs or {}).get("result", [])
        submodel_infos = []
        for ref in refs:
            try:
                sm_id = ref["keys"][0]["value"]
                submodel_infos.append({"name": sm_id.strip("/").split("/")[-1], "id": sm_id})
            except Exception:
                continue
        out.append({
            "aas_id": aas_id,
            "aas_idShort": sh.get("idShort"),
            "submodels": submodel_infos,
        })
    return out


async def _load_or_build_graph(
    endpoint: str, aas_id: str, aas_idShort: Optional[str]
) -> Optional[nx.DiGraph]:
    """Return the cached AAS knowledge graph for one shell, building it on
    first use. Graph is persisted as `<stem>_graph.json` next to the source
    AASX/JSON so a re-run reuses it without hitting the server."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_stem = aas_idShort if aas_idShort else _b64(aas_id)
    graph_cache = _RESULTS_DIR / f"{cache_stem}_graph.json"

    if graph_cache.exists():
        try:
            with open(graph_cache, "r", encoding="utf-8") as f:
                return nx.node_link_graph(json.load(f))
        except Exception:
            pass

    graph_path = None
    try:
        json_path = await aas_loader.get_json(endpoint=endpoint, aas_id=aas_id, base_dir=str(_RESULTS_DIR))
    except Exception:
        json_path = None
    if json_path and os.path.exists(json_path):
        aas_loader.aas_json_parser(json_path)
        graph_path = json_path.replace(".json", "_graph.json")

    if not graph_path or not os.path.exists(graph_path):
        try:
            aasx_path = await aas_loader.get_aasx(endpoint=endpoint, aas_id=aas_id, base_dir=str(_RESULTS_DIR))
        except Exception:
            aasx_path = None
        if aasx_path and os.path.exists(aasx_path):
            aas_loader.aasx_parser(aasx_path)
            graph_path = aasx_path.replace(".aasx", "_graph.json")

    if not graph_path or not os.path.exists(graph_path):
        return None

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            G = nx.node_link_graph(json.load(f))
    except Exception:
        return None

    try:
        with open(graph_cache, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(G), f, indent=2)
    except Exception:
        pass
    return G


def _attr_string(node_data: Dict[str, Any]) -> str:
    """Compact, search-friendly string built from a graph node's attributes."""
    parts = []
    ids = _norm(str(node_data.get("idShort", "")))
    if ids:
        parts.append(f"title: {ids}")
    cd_def = _norm(str(node_data.get("cd_definition", "")))
    desc = _norm(str(node_data.get("description", "")))
    final_desc = cd_def if cd_def else desc
    if final_desc and final_desc != "None":
        parts.append(f"description: {final_desc}")
    unit = _norm(str(node_data.get("cd_unit", "")))
    if unit:
        parts.append(f"unit: {unit}")
    return ", ".join(parts)


def _collect_candidate_nodes(G: nx.DiGraph) -> List[Tuple[str, Dict[str, Any]]]:
    """Return ranking candidates: every node *except* containers (AAS, SM, SMC, SML)."""
    out = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") in _CONTAINER_TYPES:
            continue
        out.append((node_id, data))
    return out


def _submodel_ancestor_idshort(G: nx.DiGraph, node_id: str) -> Optional[str]:
    """Walk up predecessors from `node_id` and return the `idShort` of the
    first ancestor whose type is `Submodel`. Returns None if none found.
    Result is suitable for matching against the user-supplied scope list."""
    cur = node_id
    seen = set()
    while cur not in seen:
        seen.add(cur)
        data = G.nodes[cur]
        if data.get("type") == "Submodel":
            return data.get("idShort")
        preds = list(G.predecessors(cur))
        if not preds:
            return None
        cur = preds[0]
    return None


def _filter_by_submodel_scope(
    G: nx.DiGraph,
    candidates: List[Tuple[str, Dict[str, Any]]],
    scope: Optional[List[str]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Drop candidates whose Submodel ancestor's idShort is not in `scope`.
    Empty / None scope means no filtering. Matching is case-insensitive."""
    if not scope:
        return candidates
    wanted = {s.strip().lower() for s in scope if s and s.strip()}
    if not wanted:
        return candidates
    out = []
    for node_id, data in candidates:
        sm_ids = _submodel_ancestor_idshort(G, node_id)
        if sm_ids and sm_ids.lower() in wanted:
            out.append((node_id, data))
    return out


def _entity_retrieval(
    keywords: Dict[str, List[str]],
    attr_texts: List[str],
    top_m: int,
    *,
    per_term_cap: int = 64,
) -> Tuple[List[int], List[float]]:
    """Hybrid score = 0.6 · bucket-weighted sim + 0.3 · max per-term sim + 0.1 · lexical boost.
    Buckets: process / material / parameter / objective.
    """
    if not attr_texts:
        return [], []

    emb = _embedder()
    vecs = emb.embed_documents(attr_texts)
    if not vecs:
        return [], []
    V = np.array(vecs, dtype="float32")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    def _join(xs):
        return " | ".join([x for x in (xs or []) if x])

    bucket_texts_raw = [
        ("Process:", _join(keywords.get("process_terms", []))),
        ("Material:", _join(keywords.get("material_terms", []))),
        ("Parameters:", _join(keywords.get("parameter_terms", []))),
        ("Objectives:", _join(keywords.get("objective_terms", []))),
    ]
    bucket_texts = [f"{label} {body}" for label, body in bucket_texts_raw if body]
    if bucket_texts:
        bucket_vecs = []
        for t in bucket_texts:
            qv = np.array(emb.embed_query(t), dtype="float32")
            qv /= (np.linalg.norm(qv) + 1e-12)
            bucket_vecs.append(qv)
        B = np.stack(bucket_vecs, axis=1)
        w = np.full((B.shape[1],), 1.0 / B.shape[1], dtype="float32")
        bucket_weighted = (Vn @ B * w).sum(axis=1)
    else:
        bucket_weighted = np.zeros((Vn.shape[0],), dtype="float32")

    seen, uniq = set(), []
    for k in ("process_terms", "material_terms", "parameter_terms", "objective_terms"):
        for t in keywords.get(k, []) or []:
            tt = (t or "").strip()
            if not tt or tt.lower() in seen:
                continue
            seen.add(tt.lower())
            uniq.append(tt)
            if len(uniq) >= per_term_cap:
                break
        if len(uniq) >= per_term_cap:
            break

    if uniq:
        term_vecs = []
        for t in uniq:
            qv = np.array(emb.embed_query(t), dtype="float32")
            qv /= (np.linalg.norm(qv) + 1e-12)
            term_vecs.append(qv)
        T = np.stack(term_vecs, axis=1)
        max_per_term = (Vn @ T).max(axis=1)
    else:
        max_per_term = np.zeros((Vn.shape[0],), dtype="float32")

    lowers = [t.lower() for t in uniq]
    boosts = np.zeros((len(attr_texts),), dtype="float32")
    for i, text in enumerate(attr_texts):
        hay = (text or "").lower()
        hits = sum(1 for t in lowers if t and t in hay)
        boosts[i] = min(0.10, 0.02 * hits)

    final = 0.6 * bucket_weighted + 0.3 * max_per_term + 0.1 * boosts

    k = min(top_m, len(attr_texts))
    if k <= 0:
        return [], []
    idxs = np.argpartition(-final, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-final[idxs])]
    return idxs.tolist(), final[idxs].astype(float).tolist()


def _expand_context(G: nx.DiGraph, node_id: str) -> Dict[str, Any]:
    """Use graph edges (not string-path heuristics) to attach the parent
    container and the list of sibling idShorts to a retrieved node — these
    together form the 'evidence neighbourhood' shown to the LLM."""
    out: Dict[str, Any] = {"parent": None, "siblings": []}
    if node_id not in G:
        return out
    parents = list(G.predecessors(node_id))
    if not parents:
        return out
    parent = parents[0]
    pdata = G.nodes[parent]
    out["parent"] = {
        "idShort": pdata.get("idShort"),
        "type": pdata.get("type"),
        "semantic_path": parent,
        "description": pdata.get("description"),
    }
    out["siblings"] = [
        G.nodes[sib].get("idShort", "Unknown")
        for sib in G.successors(parent)
        if sib != node_id
    ]
    return out


async def _read_values(
    endpoint: str, items: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch the live `$value` for every selected entity via the BaSyx REST API."""
    client = BasyxApiClient(endpoint)
    ok, errs = [], []
    for it in items:
        api = it.get("api_path") or ""
        if not api:
            errs.append({"entity": it, "error": "missing_api_path"})
            continue
        try:
            res = await client.get(api.rstrip("/") + "/$value")
            it2 = dict(it)
            it2["value"] = res
            ok.append(it2)
        except Exception as ex:
            errs.append({"entity": it, "error": f"read_failed: {ex!s}"})
    return ok, errs


class KnowledgeFindBuildContext(BaseTool):
    name: str = "knowledge_find_build_context"
    description: str = (
        "Given a keyword set K (process/material/parameter/objective buckets) and an AAS server "
        "endpoint S, discover printer shells, parse each into a knowledge graph, retrieve the "
        "top-m relevant entities by hybrid semantic+lexical scoring with graph-based context "
        "expansion, read their live $value endpoints, and return an evidence pack E_a. "
        "Use the optional `submodel_scope` argument to restrict retrieval to one or more AAS "
        "submodels by idShort: 'TechnicalData' for printer-suitability evidence (capabilities, "
        "specs), 'ProductionOperations' for operational-readiness evidence (current job, machine "
        "state), 'PrintRecords' for historical quality evidence (past builds, density, defect "
        "modes), or 'Nameplate' for manufacturer/identification data. Omit or pass an empty "
        "list to search across all submodels."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": (
                    "Keyword set in JSON format with fields "
                    "'process_terms', 'material_terms', 'parameter_terms', 'objective_terms'."
                ),
            },
            "endpoint": {
                "type": "string",
                "description": "AAS server base URL.",
                "default": "http://localhost:8081",
            },
            "top_m": {
                "type": "integer",
                "description": "Number of top entities to retrieve and read.",
                "default": 20,
            },
            "submodel_scope": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of submodel idShorts to restrict the search to "
                    "(e.g., ['TechnicalData'], ['PrintRecords'], or ['TechnicalData','ProductionOperations']). "
                    "Omit or leave empty to search across all submodels."
                ),
                "default": [],
            },
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        keywords: str,
        endpoint: str = "http://localhost:8081",
        top_m: int = 20,
        submodel_scope: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolResult:
        try:
            kw = _as_obj(keywords)
            buckets = {
                "process_terms": kw.get("process_terms", []),
                "material_terms": kw.get("material_terms", []),
                "parameter_terms": kw.get("parameter_terms", []),
                "objective_terms": kw.get("objective_terms", []),
            }

            shells = await _aas_explore(endpoint)
            if not shells:
                return ToolResult(error="No AAS shells found at endpoint.")

            combined = nx.DiGraph()
            per_shell_nodes: Dict[str, Tuple[str, str]] = {}
            for sh in shells:
                G = await _load_or_build_graph(endpoint, sh["aas_id"], sh.get("aas_idShort"))
                if G is None or G.number_of_nodes() == 0:
                    continue
                stem = sh.get("aas_idShort") or _b64(sh["aas_id"])
                for n, d in G.nodes(data=True):
                    nid = n if n.startswith(stem) else f"{stem}::{n}"
                    combined.add_node(nid, **d)
                    per_shell_nodes[nid] = (sh["aas_id"], sh.get("aas_idShort"))
                for u, v in G.edges():
                    up = u if u.startswith(stem) else f"{stem}::{u}"
                    vp = v if v.startswith(stem) else f"{stem}::{v}"
                    combined.add_edge(up, vp)

            if combined.number_of_nodes() == 0:
                return ToolResult(error="AAS discovered but no graph could be built.")

            candidates = _collect_candidate_nodes(combined)
            candidates = _filter_by_submodel_scope(combined, candidates, submodel_scope)
            if not candidates:
                scope_repr = submodel_scope or "all"
                return ToolResult(
                    error=f"No candidate entities under submodel_scope={scope_repr}. "
                          f"Available submodel idShorts: TechnicalData, Nameplate, "
                          f"ProductionOperations, PrintRecords."
                )
            attr_texts = [_attr_string(d) for _, d in candidates]
            sel_idx, sel_scores = _entity_retrieval(buckets, attr_texts, top_m)

            selected_items = []
            for rank, (idx, score) in enumerate(zip(sel_idx, sel_scores), start=1):
                node_id, data = candidates[idx]
                ctx = _expand_context(combined, node_id)
                aas_id, aas_idShort = per_shell_nodes.get(node_id, (None, None))
                selected_items.append({
                    "rank": rank,
                    "score": float(score),
                    "node_id": node_id,
                    "aas_idShort": aas_idShort,
                    "idShort": data.get("idShort"),
                    "type": data.get("type"),
                    "description": data.get("description"),
                    "unit": data.get("cd_unit") or None,
                    "semantic_path": node_id,
                    "api_path": data.get("API_path"),
                    "parent": ctx["parent"],
                    "siblings": ctx["siblings"],
                })

            to_read = [
                {"node_id": s["node_id"], "api_path": s["api_path"]}
                for s in selected_items if s.get("api_path")
            ]
            ok, errs = await _read_values(endpoint, to_read)
            val_by_node = {r["node_id"]: r.get("value") for r in ok}
            for s in selected_items:
                s["value"] = val_by_node.get(s["node_id"])

            evidence = {
                "meta": {
                    "timestamp": _now_iso(),
                    "endpoint": endpoint,
                    "query_keywords": buckets,
                    "top_m": top_m,
                    "submodel_scope": submodel_scope or "all",
                },
                "catalog": {
                    "considered_shells": shells,
                    "num_shells_considered": len(shells),
                    "num_entities_considered": len(candidates),
                },
                "retrieval": {"selected": selected_items},
                "errors": errs,
            }
            _save_evidence(evidence)

            summarized = [
                {
                    "entity_name": s.get("idShort"),
                    "aas_idShort": s.get("aas_idShort"),
                    "value": s.get("value"),
                    "unit": s.get("unit"),
                    "description": s.get("description"),
                    "entity_position": s.get("semantic_path"),
                    "context_parent": (s.get("parent") or {}).get("idShort"),
                    "context_siblings": s.get("siblings"),
                }
                for s in selected_items
            ]
            pretty = json.dumps(summarized, ensure_ascii=False, indent=2)
            return ToolResult(
                output="The relevant info in the printers' AAS-based Digital Twins are\n" + pretty
            )

        except Exception as e:
            return ToolResult(error=f"knowledge_find_build_context failed: {e!s}")
