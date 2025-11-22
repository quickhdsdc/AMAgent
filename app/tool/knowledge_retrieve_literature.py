from app.tool.base import BaseTool, ToolResult
import os, json, glob, re, uuid, time, pickle, random, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set, Sequence, Iterable, Optional
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss

from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# Data model
# ---------------------------

@dataclass
class Doc:
    doc_id: str
    name: str
    fileName: str
    content: str
    summary: str

# ---------------------------
# Summarization helper
# ---------------------------

def _build_plain_summary(
    content: str,
    process_terms: List[str] = None,
    material_terms: List[str] = None,
    parameter_terms: List[str] = None,
    objective_terms: List[str] = None,
    model: str = "gpt-5-2025-08-07",
) -> str:
    process_terms = process_terms or []
    material_terms = material_terms or []
    parameter_terms = parameter_terms or []
    objective_terms = objective_terms or []

    system_msg = (
        "You are an extractive technical summarizer for LPBF/SLM literature. "
        "Write ONE plain-text paragraph (no bullets, no markdown) that captures: "
        "clear cause→effect links between process parameters with explicit quantitative values"
        "(laser power, scan speed/velocity, hatch spacing, beam diameter, layer thickness, "
        "linear energy density) and outcomes (defects: keyhole, lack of fusion, porosity; "
        "geometry: melt-pool width/depth/length); and "
        "Do not invent numbers or facts; only use what is in the content. "
        "If no quantitative values are present, explicitly say: 'No quantitative parameter values found.' "
        "Be concise. No citations, no references to figures, no headings. "
        "Prioritize statements involving the provided objective terms. "
    )

    user_msg = (
        "QUERY TERMS\n"
        f"- Process: {', '.join(process_terms)}\n"
        f"- Material: {', '.join(material_terms)}\n"
        f"- Parameters: {', '.join(parameter_terms)}\n"
        f"- Objectives: {', '.join(objective_terms)}\n"
        "-----\n"
        "CONTENT\n"
        f"{content}\n"
        "-----\n"
        "Only use information from CONTENT. Output a single paragraph."
    )

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}]
    )
    return (resp.choices[0].message.content or "").strip()

# ---------------------------
# File / content helpers
# ---------------------------
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

def _sanitize_filename(name: str, max_len: int = 150) -> str:
    s = re.sub(r"[^\w\-.,\s]+", "_", name).strip()
    s = re.sub(r"\s+", "_", s)
    return (s[:max_len] or "item").rstrip("_")

def _wrap(s: str, width: int = 100) -> str:
    return textwrap.fill(s.replace("\r", " ").replace("\n", " "), width=width)

def _doc_id_of(d) -> Optional[str]:
    if hasattr(d, "doc_id") and getattr(d, "doc_id"):
        return str(getattr(d, "doc_id"))
    if hasattr(d, "id") and getattr(d, "id"):
        return str(getattr(d, "id"))
    meta = getattr(d, "metadata", None)
    if isinstance(meta, dict) and meta.get("doc_id"):
        return str(meta["doc_id"])
    return None

def _content_of_doc(d) -> Optional[str]:
    if hasattr(d, "content") and getattr(d, "content"):
        return str(getattr(d, "content"))
    if hasattr(d, "page_content") and getattr(d, "page_content"):
        return str(getattr(d, "page_content"))
    return None

def _page_text_from_doc(d, page_idx: int) -> Optional[str]:
    meta = getattr(d, "metadata", None)
    if not isinstance(meta, dict):
        return None
    for key in ("pages", "page_texts"):
        if key in meta and isinstance(meta[key], list):
            arr = meta[key]
            if 0 <= page_idx < len(arr):
                return str(arr[page_idx])
    return None

def _build_id_index(docs) -> Dict[str, Any]:
    index = {}
    for d in docs or []:
        did = _doc_id_of(d)
        if did:
            index[did] = d
    return index

def _resolve_content_from_docs(result: Dict[str, Any], id2doc: Dict[str, Any]) -> Optional[str]:
    rid = result.get("doc_id") or result.get("id")
    if not rid:
        return None
    d = id2doc.get(str(rid))
    if not d:
        return None
    if "page" in result and result["page"] is not None:
        try:
            page_idx = int(result["page"])
            ptxt = _page_text_from_doc(d, page_idx)
            if ptxt:
                return ptxt
        except Exception:
            pass
    return _content_of_doc(d)

def _pick_content(result: Dict[str, Any], id2doc: Optional[Dict[str, Any]] = None) -> str:
    for key in ("content", "text", "chunk", "snippet", "body", "chunk_text", "preview"):
        if key in result and result[key]:
            return str(result[key])
    if "passages" in result and isinstance(result["passages"], list) and result["passages"]:
        joined = "\n\n".join(map(str, result["passages"]))
        if joined.strip() and len(joined.strip()) > 40:
            return joined
    if id2doc:
        resolved = _resolve_content_from_docs(result, id2doc)
        if resolved:
            return resolved
    return ""

# ---------------------------
# Printing / saving
# ---------------------------
def print_and_save_evidence(
    evidence: Dict[str, Any],
    docs: Optional[List[Any]] = None,
    base_dir: str = "results_AM",
    run_name: str | None = None,
    max_items_preview: int = 10,
    preview_chars: int = 300,
    wrap_width: int = 100,
) -> tuple[str, List[str]]:
    results: List[Dict[str, Any]] = evidence.get("results", [])
    # ts = time.strftime("%Y%m%d_%H%M%S")
    # run_name = run_name or "bfsdfs"
    out_dir = Path(base_dir)# / f"{ts}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    id2doc = _build_id_index(docs) if docs is not None else {}

    print("\n=== Evidence Pack Summary ===")
    print(f"Results: {len(results)}")
    for r in results[:max_items_preview]:
        name = r.get("name", "<no name>")
        score = r.get("score_hybrid", r.get("score", 0.0))
        content = _pick_content(r, id2doc=id2doc)
        preview = content[:preview_chars] + (" …" if len(content) > preview_chars else "")
        print(f"- {score:.3f} :: {name}")
        print(f"  {_wrap(preview, wrap_width)}\n")
    if len(results) > max_items_preview:
        print(f"...and {len(results) - max_items_preview} more (saved to {out_dir}).")

    with open(out_dir / "evidence_full.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    md_dir = out_dir / "items"
    md_dir.mkdir(exist_ok=True)
    idx_to_mdtext: Dict[int, str] = {}

    for idx, r in enumerate(results, start=1):
        name = r.get("name", f"item_{idx}")
        score = r.get("score_hybrid", r.get("score", 0.0))
        content = _pick_content(r, id2doc=id2doc)
        if not content and "passages" in r:
            content = "\n\n".join(map(str, r["passages"])) or ""
        content = content or ""

        fname = f"{idx:04d}_{_sanitize_filename(name)}.md"
        fpath = md_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n")
            f.write(f"**Score:** {score:.3f}\n\n")
            for k in ("doc_id", "id", "source", "page", "url"):
                if k in r:
                    f.write(f"- **{k}:** `{r[k]}`\n")
            if "metadata" in r:
                f.write(f"- **metadata:** `{r['metadata']}`\n")
            f.write("\n---\n\n")
            f.write(content if content else "_<no content>_\n")
        idx_to_mdtext[idx] = content

    # ---- Build summaries (and write results.jsonl as before) ----
    summaries: List[str] = []
    with open(out_dir / "knowledge_retrieve_literature_results.jsonl", "w", encoding="utf-8") as f:
        for idx, r in enumerate(results, start=1):
            content = idx_to_mdtext.get(idx, "")
            summary_text = _build_plain_summary(
                content,
                process_terms=evidence.get("query_facets", {}).get("process_terms", []),
                material_terms=evidence.get("query_facets", {}).get("material_terms", []),
                parameter_terms=evidence.get("query_facets", {}).get("parameter_terms", []),
                objective_terms=evidence.get("query_facets", {}).get("objective_terms", []),
            )
            summaries.append(summary_text)

            out_record = {
                "doc_id": r.get("doc_id") or r.get("id"),
                "fileName": r.get("fileName"),
                "score_hybrid": r.get("score_hybrid"),
                "score_semantic": r.get("score_semantic"),
                "lexical_boost": r.get("lexical_boost"),
                "facet_matches": r.get("facet_matches", {}),
                "summary": summary_text,
                "passages": [content] if content else [],
            }
            for k in ("source", "page", "url", "metadata"):
                if k in r:
                    out_record[k] = r[k]
            json.dump(out_record, f, ensure_ascii=False)
            f.write("\n")

    # Also persist a human-readable summaries file (optional)
    with open(out_dir / "summaries.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(summaries, start=1):
            f.write(f"[{i}] {s}\n\n")

    # with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
    #     f.write(
    #         "This folder contains the retrieval output.\n"
    #         "- evidence_full.json : original evidence dict (as produced by retrieval)\n"
    #         "- results.jsonl      : revised records (no 'name', plus 'summary', 'passages'=original content)\n"
    #         "- items/*.md         : per-result markdown files with original content\n"
    #         "- summaries.txt      : plain-text summaries (one per result)\n"
    #     )

    print(f"\nSaved results to: {out_dir}")
    return str(out_dir), summaries

# ---------------------------
# Retrieval helpers
# ---------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / norms

def sent_split(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

def lower_in(text: str, term: str) -> bool:
    return term.lower() in text.lower()

def boolean_gate(doc: Doc, process_terms: List[str], material_terms: List[str]) -> Tuple[bool, Dict[str, List[str]]]:
    matched = {"process": [], "material": []}
    hay = f"{doc.name}\n{doc.summary}\n{doc.content}".lower()
    ok_p = False
    for t in process_terms:
        if t and t.lower() in hay:
            matched["process"].append(t)
            ok_p = True
    ok_m = False
    for t in material_terms:
        if t and t.lower() in hay:
            matched["material"].append(t)
            ok_m = True
    return ok_p and ok_m, matched

def build_query_text(process_terms: List[str],
                     material_terms: List[str],
                     parameter_terms: List[str],
                     objective_terms: List[str]) -> str:
    blocks = [
        "Process: " + " | ".join(process_terms),
        "Material: " + " | ".join(material_terms),
        "Parameters: " + " | ".join(parameter_terms),
    ]
    if objective_terms:
        blocks.append("Objective: " + " | ".join(objective_terms))
    return "\n".join(blocks)

# ---------------------------
# Corpus & embeddings
# ---------------------------

def load_corpus(corpus_dir: str) -> List[Doc]:
    docs: List[Doc] = []
    for path in glob.glob(os.path.join(corpus_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        for r in records:
            name = r.get("name", os.path.basename(path))
            fileName = r.get("fileName", os.path.basename(path))
            content = r.get("content", "")
            summary = r.get("summary", "")
            docs.append(Doc(
                doc_id=str(uuid.uuid4()),
                name=name,
                fileName=fileName,
                content=content,
                summary=summary
            ))
    return docs

def _chunks(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def embed_texts(
    texts: List[str],
    *,
    batch_size: int = 32,
    max_retries: int = 6,
    inter_batch_sleep: float = 0.0,
) -> np.ndarray:
    embeddings_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large-1"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    all_vecs: List[List[float]] = []
    for batch in _chunks(texts, batch_size):
        attempt = 0
        while True:
            try:
                vecs = embeddings_model.embed_documents(list(batch))
                all_vecs.extend(vecs)
                break
            except Exception as e:
                msg = str(e).lower()
                is_rate_limited = "429" in msg or "rate limit" in msg
                if not is_rate_limited or attempt >= max_retries:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                wait = max(wait, 1.0)
                print(f"[429] retry {attempt + 1}/{max_retries} in {wait:.1f}s; batch_size={len(batch)}")
                time.sleep(wait)
                attempt += 1
        if inter_batch_sleep > 0:
            time.sleep(inter_batch_sleep)
    return np.array(all_vecs, dtype="float32")

def embed_and_cache(docs, cache_path: str = "./corpus_AM/doc_vecs.pkl") -> np.ndarray:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            doc_vecs = pickle.load(f)
        print(f"Loaded embeddings from {cache_path}")
        return doc_vecs
    doc_texts = [d.content for d in docs]
    doc_vecs = embed_texts(doc_texts, batch_size=32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(doc_vecs, f)
    print(f"Saved embeddings to {cache_path}")
    return doc_vecs

def build_faiss_index(vecs: np.ndarray) -> faiss.IndexFlatIP:
    normed = normalize_rows(vecs).astype("float32")
    d = normed.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(normed)
    return index

# ---------------------------
# Missing pieces: RetrievalParams + retrieve_bfs_dfs
# ---------------------------

@dataclass
class RetrievalParams:
    k_per_layer: int = 10
    knn_neighbors_for_dfs: int = 8
    L_max: int = 4
    alpha: float = 0.7
    epsilon: float = 0.25
    overlap_thresh: float = 0.6

def frontier_overlap(prev_ids: Set[int], cur_ids: Set[int]) -> float:
    if not prev_ids and not cur_ids:
        return 0.0
    inter = len(prev_ids.intersection(cur_ids))
    denom = max(1, min(len(prev_ids), len(cur_ids)))
    return inter / denom

def lexical_boost(doc: Doc,
                  process_terms: List[str],
                  material_terms: List[str],
                  parameter_terms: List[str]) -> float:
    hay_title = f"{doc.name} {doc.summary}".lower()
    hay_all = f"{doc.name}\n{doc.summary}\n{doc.content}".lower()
    boost = 0.0
    if any(t.lower() in hay_title for t in process_terms) and any(t.lower() in hay_title for t in material_terms):
        boost += 0.08
    p_hits = sum(1 for t in parameter_terms if t.lower() in hay_all)
    boost += min(0.10, 0.02 * p_hits)
    return boost

def retrieve_bfs_dfs(
    docs: List[Doc],
    doc_vecs: np.ndarray,
    index: faiss.IndexFlatIP,
    process_terms: List[str],
    material_terms: List[str],
    parameter_terms: List[str],
    objective_terms: List[str] = None,
    params: RetrievalParams = RetrievalParams(),
) -> Dict[str, Any]:
    objective_terms = objective_terms or []
    doc_vecs_n = normalize_rows(doc_vecs).astype("float32")

    # Boolean gate cache
    allowed_mask = []
    matched_facets = []
    for d in docs:
        ok, matched = boolean_gate(d, process_terms, material_terms)
        allowed_mask.append(ok)
        matched_facets.append(matched)
    allowed_idx = np.array(allowed_mask, dtype=bool)

    # Initial query
    q_text = build_query_text(process_terms, material_terms, parameter_terms, objective_terms)
    q_vec = embed_texts([q_text])[0].astype("float32")
    q_vec = (q_vec / (np.linalg.norm(q_vec) + 1e-12)).astype("float32")

    seen: Set[int] = set()
    frontier_prev: Set[int] = set()
    trace = []
    collected: Dict[int, float] = {}

    for t in range(1, params.L_max + 1):
        layer_type = "BFS" if t % 2 == 1 else "DFS"

        if layer_type == "BFS":
            D = 5 * params.k_per_layer
            sims, idxs = index.search(q_vec.reshape(1, -1), D)
            idxs = idxs[0].tolist()
            sims = sims[0].tolist()
            pairs = [(i, s) for i, s in zip(idxs, sims) if i != -1 and allowed_idx[i] and s >= params.epsilon]
            pairs = pairs[: 3 * params.k_per_layer]
            scored = []
            for i, s in pairs:
                b = lexical_boost(docs[i], process_terms, material_terms, parameter_terms)
                scored.append((i, s + b, s, b))
            scored.sort(key=lambda x: x[1], reverse=True)
            layer = [i for (i, _, _, _) in scored[: params.k_per_layer]]
        else:
            neighbor_scores = defaultdict(float)
            frontier = list(frontier_prev) if frontier_prev else list(seen)
            if not frontier:
                frontier = list(range(min(len(docs), params.k_per_layer)))
            for i in frontier:
                sims, idxs = index.search(doc_vecs_n[i].reshape(1, -1), params.knn_neighbors_for_dfs + 1)
                idxs = idxs[0].tolist()
                sims = sims[0].tolist()
                for j, s in zip(idxs, sims):
                    if j == -1 or j == i:
                        continue
                    if not allowed_idx[j] or s < params.epsilon:
                        continue
                    neighbor_scores[j] = max(neighbor_scores[j], s)
            scored = []
            for j, s in neighbor_scores.items():
                sim_q = cosine_sim(q_vec, doc_vecs_n[j])
                b = lexical_boost(docs[j], process_terms, material_terms, parameter_terms)
                scored.append((j, 0.5 * sim_q + 0.5 * s + b, sim_q, s, b))
            scored.sort(key=lambda x: x[1], reverse=True)
            layer = [j for (j, *_rest) in scored[: params.k_per_layer]]

        frontier_cur = set(layer)
        ov = frontier_overlap(frontier_prev, frontier_cur)

        for i in layer:
            seen.add(i)
            sim_q = cosine_sim(q_vec, doc_vecs_n[i])
            b = lexical_boost(docs[i], process_terms, material_terms, parameter_terms)
            collected[i] = max(collected.get(i, 0.0), sim_q + b)

        if layer:
            centroid = normalize_rows(doc_vecs_n[layer, :].mean(axis=0, keepdims=True))[0]
            q_vec = (params.alpha * q_vec + (1.0 - params.alpha) * centroid).astype("float32")
            q_vec = (q_vec / (np.linalg.norm(q_vec) + 1e-12)).astype("float32")

        trace.append({
            "t": t,
            "layer": layer_type,
            "frontier_size": len(frontier_cur),
            "overlap_with_prev": ov,
        })

        if ov >= params.overlap_thresh:
            break
        frontier_prev = frontier_cur

    final = []
    for i, _prev in collected.items():
        sim_q = cosine_sim(q_vec, doc_vecs_n[i])
        b = lexical_boost(docs[i], process_terms, material_terms, parameter_terms)
        final.append((i, sim_q + b, sim_q, b))
    final.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score, sim_q, boost in final:
        ok, matched = boolean_gate(docs[i], process_terms, material_terms)
        spans = []
        if docs[i].summary:
            spans.extend(sent_split(docs[i].summary)[:2])
        if not spans and docs[i].content:
            spans.extend(sent_split(docs[i].content)[:2])
        results.append({
            "doc_id": docs[i].doc_id,
            "name": docs[i].name,
            "fileName": docs[i].fileName,
            "score_hybrid": round(score, 4),
            "score_semantic": round(sim_q, 4),
            "lexical_boost": round(boost, 4),
            "facet_matches": matched,
            "passages": spans,
        })

    evidence_pack = {
        "query_facets": {
            "process_terms": process_terms,
            "material_terms": material_terms,
            "parameter_terms": parameter_terms,
            "objective_terms": objective_terms,
        },
        "retrieval_params": {
            "k_per_layer": params.k_per_layer,
            "knn_neighbors_for_dfs": params.knn_neighbors_for_dfs,
            "L_max": params.L_max,
            "alpha": params.alpha,
            "epsilon": params.epsilon,
            "overlap_thresh": params.overlap_thresh,
        },
        "trace": trace,
        "results": results,
    }
    return evidence_pack

# ---------------------------
# MCP Tool
# ---------------------------
class KnowledgeRetrievalLiterature(BaseTool):
    name: str = "knowledge_retrieve_literature"
    description: str = (
        "Given a keyword set produced by task_extract_keywords, run literature retrieval to find evidence linking process parameters to print outcomes."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "Keyword set in JSON schema, containing process_terms, material_terms, parameter_terms, and objective_terms.",
            }
        },
        "required": ["keywords"]
    }

    async def execute(self,keywords: str, **kwargs) -> ToolResult:
        try:
            #keywords = '{"process_terms": ["LPBF", "Laser Powder Bed Fusion", "Selective Laser Melting", "Powder Bed Fusion", "SLM"], "material_terms": ["IN625", "Inconel 625", "Alloy 625"], "parameter_terms": ["laser power", "power", "scan velocity", "scan speed", "beam diameter", "spot size", "laser spot size", "layer thickness", "powder layer thickness", "hatch spacing", "hatch distance"], "objective_terms": ["keyhole", "lack of fusion", "porosity", "melt pool depth", "melt pool width", "melt pool length"]}'
            data = _as_obj(keywords)
            process_terms = data.get("process_terms", [])
            material_terms = data.get("material_terms", [])
            parameter_terms = data.get("parameter_terms", [])
            objective_terms = data.get("objective_terms", [])

            # Internal defaults
            corpus_dir = "./corpus_AM"
            cache_path = "./corpus_AM/doc_vecs.pkl"
            output_base_dir = "results_AM"
            run_name = "BFS_DFS"
            max_items_preview = 10
            preview_chars = 300
            wrap_width = 100

            # Retrieval knobs
            params_obj = RetrievalParams(
                k_per_layer=10,
                knn_neighbors_for_dfs=8,
                L_max=4,
                alpha=0.7,
                epsilon=0.25,
                overlap_thresh=0.6,
            )

            objective_terms = objective_terms or []
            # Pipeline
            docs = load_corpus(corpus_dir)
            if not docs:
                return ToolResult(error=f"No JSON files found in corpus_dir='{corpus_dir}'.")

            doc_vecs = embed_and_cache(docs, cache_path=cache_path)
            index = build_faiss_index(doc_vecs)

            evidence = retrieve_bfs_dfs(
                docs, doc_vecs, index,
                process_terms=process_terms,
                material_terms=material_terms,
                parameter_terms=parameter_terms,
                objective_terms=objective_terms,
                params=params_obj
            )

            out_dir, summaries = print_and_save_evidence(
                evidence,
                docs=docs,
                base_dir=output_base_dir,
                run_name=run_name,
                max_items_preview=max_items_preview,
                preview_chars=preview_chars,
                wrap_width=wrap_width,
            )

            numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(summaries[:5], start=1))
            return ToolResult(output=numbered)

        except Exception as e:
            msg = str(e)
            hint = None
            if "AZURE_OPENAI" in msg or "authentication" in msg.lower() or "invalid url" in msg.lower():
                hint = (
                    "Verify Azure OpenAI env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                    "AZURE_OPENAI_API_VERSION, AZURE_EMBEDDINGS_DEPLOYMENT."
                )
            return ToolResult(error=f"retrieval_bfs_dfs failed: {msg}" + (f" | {hint}" if hint else ""))


