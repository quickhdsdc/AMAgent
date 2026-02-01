from app.tool.base import BaseTool, ToolResult
from app.aas_utils.basyx_client import BasyxApiClient
from app.aas_utils import aas_loader

import os
import re
import json
import time
import base64
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from langchain_openai import AzureOpenAIEmbeddings


# ----------------------------
# Helpers
# ----------------------------
def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _as_obj(text: str) -> Any:
    """
    Finds and parses the first complete JSON value in `text`.
    Handles objects `{...}` or arrays `[...]`, ignores junk around it,
    and copes with code fences. Raises ValueError if none found.
    """
    s = _strip_fences(text)
    decoder = json.JSONDecoder()
    i = 0
    n = len(s)

    # scan to the first plausible JSON start
    while i < n:
        # skip whitespace
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] in "{[":
            try:
                obj, end = decoder.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
                # not a complete JSON at this position; advance one char
                i += 1
                continue
        i += 1
    raise ValueError("No complete JSON object/array found in input.")

_EXCLUDED_TYPES = {"SubmodelElementCollection", "AssetAdministrationShell", "Submodel"}
_EXCLUDED_TYPES_L = {t.lower() for t in _EXCLUDED_TYPES}
def _filter_entities_type(entities: list[dict]) -> list[dict]:
    def get_type(e: dict) -> str:
        # common places where type shows up
        t = e.get("type") or e.get("modelType") or (e.get("modelType", {}) or {}).get("name")
        return str(t or "").strip().lower()
    out = [e for e in entities if get_type(e) not in _EXCLUDED_TYPES_L]
    return out

def _save_evidence(evidence: dict, base_dir: str = "./results_AM", filename: str = "knowledge_build_context.json") -> str:

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    out_path = base / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    return str(out_path)

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _contains_any(hay: str, needles: List[str]) -> bool:
    hay_l = hay.lower()
    return any((n or "").lower() in hay_l for n in needles if n)

def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode()

def _attr_string(row: pd.Series) -> str:
    """
    BuildAttributeString(e): compact, search-friendly string.
    """
    id_short = _norm(str(row.get("idShort", "")))
    description = _norm(str(row.get("description", "")))
    semantic_path = _norm(str(row.get("semantic_path", "")))
    parts = [
        f"title: {id_short}" if id_short else "",
        f"description: {description}" if description else "",
        f"position: {semantic_path}" if semantic_path else "",
    ]
    return ", ".join([p for p in parts if p])

def _semantic_path(row: pd.Series) -> str:
    """
    BuildSemanticPath(e): prefer precomputed column, else fallback.
    """
    sp = _norm(str(row.get("semantic_path", "")))
    if sp:
        return sp
    # Minimal fallback if not present: try stitching idShorts (if columns exist)
    # This keeps behavior predictable even with sparse CSVs.
    chain = [str(row.get("submodel", "")).strip(), str(row.get("collection", "")).strip(), str(row.get("idShort", "")).strip()]
    chain = [c for c in chain if c]
    return "/".join(chain)

def _api_path(row: pd.Series) -> str:
    """
    BuildAPIPath(e): use 'API_path' if present; else derive from semantic path + AAS rules.
    """
    api = _norm(str(row.get("API_path", "")))
    if api:
        return api
    # Conservative fallback: many parsers export 'API_path'; if missing, we can't reliably compose.
    # Return empty to signal we couldn't build a path; the caller will skip reading.
    return ""

async def _load_or_parse_aas(endpoint: str, aas_id: str, aas_idShort: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Parse and cache to ./temp/<idShort or b64(id)>.csv; reuse cache if present.
    """
    Path("./results_AM").mkdir(exist_ok=True)
    # Choose a stable cache stem
    cache_stem = aas_idShort if aas_idShort else _b64(aas_id)
    cache_csv = os.path.join("./results_AM", f"{cache_stem}.csv")

    if os.path.exists(cache_csv):
        try:
            return pd.read_csv(cache_csv)
        except Exception:
            pass

    # Try AASX
    aasx_filepath = None
    try:
        aasx_filepath = await aas_loader.get_aasx(endpoint=endpoint, aas_id=aas_id, base_dir="results_AM")
    except Exception:
        aasx_filepath = None

    if aasx_filepath and os.path.exists(aasx_filepath):
        df = aas_loader.aasx_parser(aasx_filepath)
        try:
            df.to_csv(cache_csv, index=False)
        except Exception:
            pass
        return df

    # Try JSON
    try:
        aas_json_filepath = await aas_loader.get_json(endpoint=endpoint, aas_id=aas_id, base_dir="temp")
    except Exception:
        aas_json_filepath = None

    if aas_json_filepath and os.path.exists(aas_json_filepath):
        df = aas_loader.aas_json_parser(aas_json_filepath)
        try:
            df.to_csv(cache_csv, index=False)
        except Exception:
            pass
        return df

    return None

def _embedder():
    # Use Azure OpenAI embeddings; rely on env creds.
    # Model/deployment can be adjusted here centrally.
    return AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDINGS_MODEL", "text-embedding-3-large"),
        azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large-1"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

def _entity_retrieval(
    keywords: Dict[str, List[str]],
    attr_texts: List[str],   # e.g., [e["attr_str"], ...]
    top_m: int,
    *,
    per_term_cap: int = 64,  # cap individual term embeddings for efficiency
) -> Tuple[List[int], List[float]]:
    """
    Hybrid multi-vector retrieval:
      score = 0.6 * bucket_weighted_sim + 0.3 * max_per_term_sim + 0.1 * lexical_boost
    Buckets: process_terms, material_terms, parameter_terms, objective_terms
    """
    if not attr_texts:
        return [], []
    # 1) Embed entity texts
    emb = _embedder()
    vecs = emb.embed_documents(attr_texts)
    if not vecs:
        return [], []
    V = np.array(vecs, dtype="float32")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)  # normalize for cosine

    # 2) Build 4 bucket queries
    def _join(xs): return " | ".join([x for x in (xs or []) if x])
    b_proc = _join(keywords.get("process_terms", []))
    b_mat  = _join(keywords.get("material_terms", []))
    b_par  = _join(keywords.get("parameter_terms", []))
    b_obj  = _join(keywords.get("objective_terms", []))

    # Use labeled prompts to anchor semantics a bit better
    bucket_texts = [
        f"Process: {b_proc}" if b_proc else "",
        f"Material: {b_mat}" if b_mat else "",
        f"Parameters: {b_par}" if b_par else "",
        f"Objectives: {b_obj}" if b_obj else "",
    ]
    bucket_texts = [t for t in bucket_texts if t]

    bucket_vecs = []
    for t in bucket_texts:
        qv = np.array(emb.embed_query(t), dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-12)
        bucket_vecs.append(qv)
    # Weighted bucket blend (tuneable)
    # Map weights in the same order we added bucket_texts
    # Rebalanced per user request to better find "records" (less bias to parameters)
    w_map = {"Process:":0.25, "Material:":0.25, "Parameters:":0.25, "Objectives:":0.25}
    bucket_weights = []
    for t in bucket_texts:
        key = t.split(":")[0] + ":"  # "Process:" etc.
        bucket_weights.append(w_map.get(key, 0.0))
    if bucket_weights:
        w = np.array(bucket_weights, dtype="float32")
        w = w / (w.sum() + 1e-12)
    else:
        w = np.zeros((0,), dtype="float32")

    # 3) Per-term queries (dedupe + cap)
    all_terms = []
    for k in ("process_terms", "material_terms", "parameter_terms", "objective_terms"):
        all_terms.extend(keywords.get(k, []))
    seen = set()
    uniq_terms = []
    for t in all_terms:
        tt = (t or "").strip()
        if not tt or tt.lower() in seen:
            continue
        seen.add(tt.lower())
        uniq_terms.append(tt)
        if len(uniq_terms) >= per_term_cap:
            break

    term_vecs = []
    for t in uniq_terms:
        qv = np.array(emb.embed_query(t), dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-12)
        term_vecs.append(qv)

    # 4) Compute similarities in vectorized form
    # Bucket sims: for each bucket q, sim = Vn @ q
    if bucket_vecs:
        B = np.stack(bucket_vecs, axis=1)             # [D, B]
        bucket_sims = Vn @ B                           # [N, B]
        bucket_weighted = (bucket_sims * w).sum(axis=1)  # [N]
    else:
        bucket_weighted = np.zeros((Vn.shape[0],), dtype="float32")

    # Per-term max sim
    if term_vecs:
        T = np.stack(term_vecs, axis=1)   # [D, T]
        term_sims = Vn @ T                # [N, T]
        max_per_term = term_sims.max(axis=1)  # [N]
    else:
        max_per_term = np.zeros((Vn.shape[0],), dtype="float32")

    # 5) Lexical boost (cheap)
    # Count literal hits across all terms with small saturation
    lowers = [t.lower() for t in uniq_terms]
    boosts = np.zeros((len(attr_texts),), dtype="float32")
    for i, text in enumerate(attr_texts):
        hay = (text or "").lower()
        hits = sum(1 for t in lowers if t and t in hay)
        # cap boost; tuneables: 0.02 per hit, cap 0.10
        boosts[i] = min(0.10, 0.02 * hits)

    # 6) Final score blend (tune as needed)
    final = 0.6 * bucket_weighted + 0.3 * max_per_term + 0.1 * boosts

    # 7) Top-k
    k = min(top_m, len(attr_texts))
    idxs = np.argpartition(-final, kth=k-1)[:k]
    # sort by score desc
    idxs = idxs[np.argsort(-final[idxs])]
    scores = final[idxs].astype(float).tolist()
    return idxs.tolist(), scores

async def _aas_explore(endpoint: str):

    client = BasyxApiClient(endpoint)
    shells = await client.get_shells()
    if not isinstance(shells, list):
        return []
    results = []
    for sh in shells:
        aas_id = sh.get("id")
        id_short = sh.get("idShort")
        b64_id = base64.urlsafe_b64encode(aas_id.encode()).decode()

        submodel_refs = await client.get(f"/shells/{b64_id}/submodel-refs")
        submodel_refs = submodel_refs.get('result', [])
        submodel_infos = []
        for ref in submodel_refs:
            submodel_id = ref["keys"][0]["value"]
            submodel_name = submodel_id.strip("/").split("/")[-1]  # "TechnicalData"
            submodel_infos.append({
                "name": submodel_name,
                "id": submodel_id,
            })

        results.append({
            "aas_id": aas_id,
            "aas_idShort": id_short,
            "submodels": submodel_infos,
        })

    return results

async def _parse_entities(endpoint: str, shells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ParseAAS for each matched shell, return a flat list of row dicts with paths.
    """
    entities: List[Dict[str, Any]] = []
    for sh in shells:
        aas_id = sh["aas_id"]
        aas_idShort = sh.get("aas_idShort")
        df = await _load_or_parse_aas(endpoint, aas_id, aas_idShort)
        if df is None or df.empty:
            continue
        # Expect columns: idShort, description, semantic_path, API_path (others are tolerated)
        for _, row in df.iterrows():
            id_short = _norm(str(row.get("idShort", "")))
            desc = _norm(str(row.get("description", "")))
            entity_type = _norm(str(row.get("type", "")))
            sem_path = _semantic_path(row)
            api = _api_path(row)
            entities.append({
                "idShort": id_short,
                "type": entity_type,
                "description": desc,
                "semantic_path": sem_path,
                "api_path": api,
                "attr_str": _attr_string(row),
            })
    return entities

async def _read_values(endpoint: str, entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    ReadAASValues for each entity (api_path + '/$value').
    Returns (successful_entities, errors)
    """
    client = BasyxApiClient(endpoint)
    successes = []
    errors = []
    for e in entities:
        api = e.get("api_path") or ""
        if not api:
            errors.append({"entity": e, "error": "missing_api_path"})
            continue
        value_api = api.rstrip("/") + "/$value"
        try:
            res = await client.get(value_api)
            e2 = dict(e)
            e2["value"] = res
            successes.append(e2)
        except Exception as ex:
            errors.append({"entity": e, "error": f"read_failed: {str(ex)}"})
    return successes, errors

def _build_evidence_pack(
    endpoint: str,
    keywords: List[str],
    top_m: int,
    considered_shells: List[Dict[str, Any]],
    candidate_entities: List[Dict[str, Any]],
    selected_indices: List[int],
    selected_scores: List[float],
    read_successes: List[Dict[str, Any]],
    read_errors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Map selected by index to entities
    selected_items = []
    for rank, (idx, score) in enumerate(zip(selected_indices, selected_scores), start=1):
        if 0 <= idx < len(candidate_entities):
            ent = candidate_entities[idx]
            # find read result if any
            value = None
            for s in read_successes:
                if s.get("api_path") == ent.get("api_path"):
                    value = s.get("value")
                    break
            selected_items.append({
                "rank": rank,
                "score": float(score),
                "idShort": ent.get("idShort"),
                "description": ent.get("description"),
                "semantic_path": ent.get("semantic_path"),
                "api_path": ent.get("api_path"),
                "value": value,
            })

    return {
        "meta": {
            "timestamp": _now_iso(),
            "endpoint": endpoint,
            "query_keywords": keywords,
            "top_m": top_m,
        },
        "catalog": {
            "considered_shells": considered_shells,
            "num_shells_considered": len(considered_shells),
        },
        "retrieval": {
            "num_entities_considered": len(candidate_entities),
            "selected": selected_items,
        },
        "errors": read_errors,  # include read failures or missing API paths
    }


# ----------------------------
# MCP Tool
# ----------------------------

class KnowledgeFindBuildContext(BaseTool):
    name: str = "knowledge_find_build_context"
    description: str = (
        "Given a keyword set K and an AAS endpoint S, discover AAS, parse entities, retrieve top-m relevant "
        "properties by semantic similarity, read their values, and return an evidence pack E_a."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "Keyword set in a json format with field 'process_terms','material_terms','parameter_terms','objective_terms'",
            },
            "endpoint": {
                "type": "string",
                "description": "AAS server base URL.",
                "default": "http://localhost:8081",
            },
            "top_m": {
                "type": "integer",
                "description": "Number of top entities to retrieve/read.",
                "default": 20,
            },
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }

    async def execute(self, keywords: str, endpoint: str = "http://localhost:8081", top_m: int = 20, **kwargs) -> ToolResult:
        # keywords = '{"process_terms": ["LPBF", "Laser Powder Bed Fusion", "Selective Laser Melting", "Powder Bed Fusion", "SLM"], "material_terms": ["IN625", "Inconel 625", "Alloy 625"], "parameter_terms": ["laser power", "power", "scan velocity", "scan speed", "beam diameter", "spot size", "laser spot size", "layer thickness", "powder layer thickness", "hatch spacing", "hatch distance"], "objective_terms": ["keyhole", "lack of fusion", "porosity", "melt pool depth", "melt pool width", "melt pool length"]}'
        # endpoint = "http://localhost:8081"
        # top_m = 20
        data = _as_obj(keywords)  # dict with 4 arrays
        process_terms = data.get("process_terms", [])
        material_terms = data.get("material_terms", [])
        parameter_terms = data.get("parameter_terms", [])
        objective_terms = data.get("objective_terms", [])

        try:
            # Discover
            shells = await _aas_explore(endpoint)
            if not shells:
                return ToolResult(error="No AAS shells found at endpoint.")

            # Parse AAS and build per-entity strings/paths
            entities = await _parse_entities(endpoint, shells)
            entities = _filter_entities_type(entities)
            
            # --- DEBUG: Save parsed entities ---
            try:
                debug_path = Path("./results_AM/debug_aas_entities.json")
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(entities, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to save debug entities: {e}")
            # -----------------------------------

            rows_for_embed = [(e["attr_str"]) for e in entities]
            sel_idx, sel_scores = _entity_retrieval(
                keywords={
                    "process_terms": process_terms,
                    "material_terms": material_terms,
                    "parameter_terms": parameter_terms,
                    "objective_terms": objective_terms,
                },
                attr_texts=rows_for_embed,
                top_m=top_m,
                per_term_cap=64
            )
            selected_entities = [entities[i] for i in sel_idx]

            # 6) Read values for selected
            read_successes, read_errors = await _read_values(endpoint, selected_entities)

            # 7) Build evidence pack
            evidence = _build_evidence_pack(
                endpoint=endpoint,
                keywords=keywords,
                top_m=top_m,
                considered_shells=shells,
                candidate_entities=entities,
                selected_indices=sel_idx,
                selected_scores=sel_scores,
                read_successes=read_successes,
                read_errors=read_errors,
            )
            _save_evidence(evidence)

            # 7) Return ONLY selected info in the requested shape
            selected = evidence.get("retrieval", {}).get("selected", [])
            summarized = []
            for s in selected:
                summarized.append({
                    "entity_name": s.get("idShort"),
                    "aas_idShort": s.get("aas_idShort"),
                    "value": s.get("value"),
                    "description": s.get("description"),
                    "entity_position": s.get("semantic_path"),
                })

            pretty = json.dumps(summarized, ensure_ascii=False, indent=2)
            return ToolResult(output="The relevant info in the printers' AAS-based Digital Twins are\n" + pretty)


        except Exception as e:
            return ToolResult(error=f"knowledge_find_build_context failed: {str(e)}")





