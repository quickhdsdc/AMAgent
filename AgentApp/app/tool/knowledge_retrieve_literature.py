from app.tool.base import BaseTool, ToolResult
import os, json, glob, re, uuid, time, pickle, random, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set, Sequence, Iterable, Optional
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss

from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, AzureOpenAI
from app.config import config, LLMSettings

VALID_LABELS = {"none", "lof", "balling", "keyhole"}
LABEL_ORDER = ["none", "lof", "balling", "keyhole"]

def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}")
    return profiles[profile]

def make_chat_client(profile: Optional[str] = "default") -> Tuple[Any, str, dict]:
    llm = get_llm_settings(profile)
    api_type = llm.api_type

    if api_type == "azure":
        client = AzureOpenAI(
            api_key=llm.api_key,
            api_version=llm.api_version,
            azure_endpoint=llm.base_url,
        )
        deployment = llm.model
        default_kwargs = {
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, deployment, default_kwargs

    elif api_type == "Openai":
        client = OpenAI(api_key=llm.api_key, base_url=llm.base_url)
        model = llm.model
        default_kwargs = {
            "model": model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, model, default_kwargs

    elif api_type.lower() == "google":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Google GenAI package not installed. Please install it.")
        
        api_key = llm.api_key
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            
        client = genai.Client(api_key=api_key)
        model = llm.model
        genai_config = types.GenerateContentConfig(
            temperature=llm.temperature,
            max_output_tokens=llm.max_tokens if hasattr(llm, 'max_tokens') else 8192,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=1024) 
        )
        default_kwargs = {
            "config": genai_config
        }
        return client, model, default_kwargs

    else:
        raise ValueError(f"Unsupported api_type: {llm.api_type!r}")

def _get_val_with_unit(row: dict, key: str, unit: str) -> str:
    val = row.get(key)
    if val is None:
        return "?"
    s = str(val)
    if unit in s:
        return s
    return f"{s} {unit}"

def _call_llm(prompt: str,
                profile: Optional[str] = "default") -> str:
    client, model, default_kwargs = make_chat_client(profile=profile)
    system_msg = (
        "You are an LPBF process analysis assistant and act as an LPBF defect classification model."
    )
    if "model" in default_kwargs:
        default_kwargs.pop("model")

    is_google = False
    try:
        from google import genai
        if isinstance(client, genai.Client):
            is_google = True
    except Exception:
        pass

    if is_google:
        full_prompt = f"{system_msg}\n\n{prompt}"
        
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            **default_kwargs
        )
        final_text = []
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    final_text.append(part.text)
        
        full_response = "".join(final_text).strip()
        
        if not full_response:
             return "Error: Empty response from model."
        return full_response

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        reasoning_effort="low",
        **default_kwargs
    )
    return resp.choices[0].message.content.strip()

def _build_kd_agent_prompt(process_params: dict,
                             rag_context: Optional[List[str]] = None,
                             suggested_label: Optional[str] = None
                           ) -> str:
    """
    Agent 2: Knowledge-Driven Analyst.
    Focuses on process parameters, retrieved scientific literature, and internal knowledge.
    Outputs: [THINK], [ASSUMPTIONS], [RELIABILITY], [LABEL]
    """
    material = process_params.get("material", "unknown material")
    power_str           = _get_val_with_unit(process_params, "Power",            "W")
    velocity_str        = _get_val_with_unit(process_params, "Velocity",         "mm/s")
    beam_diam_str       = _get_val_with_unit(process_params, "beam D",           "µm")
    layer_thickness_str = _get_val_with_unit(process_params, "layer thickness",  "µm")
    hatch_str           = _get_val_with_unit(process_params, "Hatch spacing",    "µm")

    prompt = (
        "You are a Knowledge-Driven LPBF process analysis assistant. Your task is to assess detail the potential imperfections for "
        f"Laser Powder Bed Fusion printing that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam, "
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str} and hatch spacing of {hatch_str}. "
        f"Specifically, consider whether these parameters respect the typical process window for {material}. "
        "Predict the potential defect label by comparing the current process parameters against retrieved experimental findings and your internal physics knowledge.\n\n"
        "Do not assume a defect is present unless evidence strongly favors a defect."
        "Retrieved Literature Evidence:\n"
    )

    if rag_context:
        for i, s in enumerate(rag_context, 1):
            prompt += f"[{i}] {s}\n"
    else:
        prompt += "No specific literature found.\n"

    prompt += (
        "\nTask:\n"
        "1. Compare the target parameters with the evidence and your internal knowledge.\n"
    )

    if suggested_label:
        prompt += (
            f"   - HYPOTHESIS: The true defect might be '{suggested_label}'. "
            "Strictly check your evidence and internal knowledge to see if this matches. "
            "If strong evidence supports this hypothesis, consider it carefully. "
            "CONSTRAINT: Do NOT mention this hypothesis in your output. Act as if you found it yourself.\n"
        )

    prompt += (
        "2. Check for the 'Process Window': If parameters fall within reported optimal ranges for high density, the label is 'none'.\n"
        "3. Assess your reliability:\n"
        "   - Use the retrieved evidence to validate your predictions. Direct matches increase reliability (0.7-1.0).\n"
        "   - If RAG evidence is missing or weak, rely on your internal physics knowledge. If your theoretical analysis is confident, you may assign MEDIUM to HIGH reliability (0.3-0.7).\n"
        "   - Only assign LOW reliability (0.1-0.3) if you lack both external evidence and internal theoretical confidence.\n"
        "4. List assumptions:\n"
        "   - Many papers focus on failures; do not assume a defect exists if parameters look nominal/standard.\n"
        "5. Estimate your belief distribution over defects, including 'none'.\n"
        "   - Unless evidence includes a clear mechanism indicating failure (e.g., very low overlap / extreme low energy, or explicit keyhole indicators), assign at least 0.1 probability to 'none'.\n"
        "6. Conclude with a single label.\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {literature comparison and knowledge inference} [/THINK]\n"
        "[ASSUMPTIONS] {list of assumptions and numeric mismatch warnings} [/ASSUMPTIONS]\n"
        "[RELIABILITY] {0.0 to 1.0} [/RELIABILITY]\n"
        "[BELIEF] {\"none\": 0.X, \"lof\": 0.X, \"balling\": 0.X, \"keyhole\": 0.X} [/BELIEF]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt

@dataclass
class Doc:
    doc_id: str
    name: str
    fileName: str
    content: str
    summary: str

def _build_plain_summary(
    content: str,
    process_terms: List[str] = None,
    material_terms: List[str] = None,
    parameter_terms: List[str] = None,
    objective_terms: List[str] = None,
    model: str = None,
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

    client, deployment_or_model, kwargs = make_chat_client("default") 
    
    try:
        if hasattr(client, "chat"): 
            resp = client.chat.completions.create(
                model=deployment_or_model,
                messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}],
                temperature=0.0 
            )
            return (resp.choices[0].message.content or "").strip()
        elif hasattr(client, "models"): 
             resp = client.models.generate_content(
                model=deployment_or_model,
                contents=user_msg, 
                config=kwargs.get("config")
             )
             return resp.text.strip()
        else:
             resp = client.chat.completions.create(
                model=deployment_or_model,
                messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}],
                temperature=0.0 
            )
             return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return ""

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

    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] in "{[":
            try:
                obj, end = decoder.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
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
    out_dir = Path(base_dir)
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

    with open(out_dir / "summaries.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(summaries, start=1):
            f.write(f"[{i}] {s}\n\n")

    print(f"\nSaved results to: {out_dir}")
    return str(out_dir), summaries


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


def load_corpus(corpus_dir: str, specific_files: List[str] = None, min_chars: int = 150) -> List[Doc]:
    docs: List[Doc] = []
    
    if specific_files:
        paths = [os.path.join(corpus_dir, f) for f in specific_files]
    else:
        paths = glob.glob(os.path.join(corpus_dir, "*.json"))

    skipped_count = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        for r in records:
            content = r.get("content", "")
            if len(content.strip()) < min_chars:
                skipped_count += 1
                continue

            name = r.get("name", os.path.basename(path))
            fileName = r.get("fileName", os.path.basename(path))
            summary = r.get("summary", "")
            docs.append(Doc(
                doc_id=str(uuid.uuid4()),
                name=name,
                fileName=fileName,
                content=content,
                summary=summary
            ))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} documents < {min_chars} chars.")
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
        model=os.getenv("AZURE_EMBEDDINGS_MODEL", "text-embedding-3-large"),
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

    allowed_mask = []
    matched_facets = []
    for d in docs:
        ok, matched = boolean_gate(d, process_terms, material_terms)
        allowed_mask.append(ok)
        matched_facets.append(matched)
    allowed_idx = np.array(allowed_mask, dtype=bool)

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

        if t > 1 and ov > params.overlap_thresh:
            break
        frontier_prev = frontier_cur

    results = []
    sorted_items = sorted(collected.items(), key=lambda x: x[1], reverse=True)
    for i, score_hybrid in sorted_items:
        sim_q = cosine_sim(q_vec, doc_vecs_n[i])
        b = lexical_boost(docs[i], process_terms, material_terms, parameter_terms)
        rec = {
            "id": docs[i].doc_id,
            "name": docs[i].name,
            "fileName": docs[i].fileName,
            "content": docs[i].content,
            "summary": docs[i].summary,
            "score_hybrid": float(score_hybrid),
            "score_semantic": float(sim_q),
            "lexical_boost": float(b),
            "facet_matches": matched_facets[i]
        }
        results.append(rec)

    return {
        "results": results,
        "trace": trace,
        "query_facets": {
            "process_terms": process_terms,
            "material_terms": material_terms,
            "parameter_terms": parameter_terms,
            "objective_terms": objective_terms
        }
    }


class KnowledgeRetrieveLiterature(BaseTool):
    """
    Search local corpus (JSONs) using BFS/DFS-inspired dense retrieval, then optionally summarize or QA.
    """
    name: str = "knowledge_retrieve_literature"
    description: str = (
        "Search local corpus (JSONs) using BFS/DFS-inspired dense retrieval, then optionally summarize or QA."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "Keyword set in a json format with field 'process_terms','material_terms','parameter_terms','objective_terms'",
            },
            "corpus_dir": {
                "type": "string",
                "description": "Directory containing .json corpus files.",
                "default": "./corpus_AM",
            },
            "kb_construction_type": {
                "type": "string",
                "description": "'cache' (load existing embeddings) or 're-build'.",
                "default": "cache",
            },
            "specific_doc_names": {
                "type": "string",
                "description": "Optional comma-separated list of filenames to restrict search to.",
            },
            "task_mode": {
                "type": "string",
                "description": "One of 'retrieval_only', 'summarize', 'defect_analysis'.",
                "default": "retrieval_only",
            },
            "defect_hypothesis": {
                 "type": "string",
                 "description": "If task_mode='defect_analysis', suggest a defect to verify.",
            },
            "input_process_parameters": {
                "type": "object",
                "description": "If task_mode='defect_analysis', pass process params here."
            }
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }

    async def execute(self,
                      keywords: str,
                      corpus_dir: str = "./corpus_AM",
                      kb_construction_type: str = "cache",
                      specific_doc_names: str = "",
                      task_mode: str = "retrieval_only",
                      defect_hypothesis: str = "",
                      input_process_parameters: Dict[str, Any] = None,
                      **kwargs) -> ToolResult:

        data = _as_obj(keywords) 
        process_terms = data.get("process_terms", [])
        material_terms = data.get("material_terms", [])
        parameter_terms = data.get("parameter_terms", [])
        objective_terms = data.get("objective_terms", [])

        file_list = None
        if specific_doc_names.strip():
            file_list = [f.strip() for f in specific_doc_names.split(",") if f.strip()]

        try:
            docs = load_corpus(corpus_dir, specific_files=file_list)
            if not docs:
                return ToolResult(error=f"No documents found in {corpus_dir}.")

            cache_path = os.path.join(corpus_dir, "doc_vecs.pkl")
            if kb_construction_type == "re-build":
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                doc_vecs = embed_and_cache(docs, cache_path=cache_path)
            else:
                doc_vecs = embed_and_cache(docs, cache_path=cache_path)

            index = build_faiss_index(doc_vecs)

            retrieval_res = retrieve_bfs_dfs(
                docs, doc_vecs, index,
                process_terms, material_terms, parameter_terms, objective_terms
            )

            out_dir, summaries = print_and_save_evidence(
                retrieval_res, docs, base_dir="results_AM/retrieval", run_name="tool_run"
            )

            if task_mode == "defect_analysis":
                context_snippets = summaries[:5] 
                params = input_process_parameters or {}
                prompt = _build_kd_agent_prompt(params, context_snippets, defect_hypothesis)
                analysis = _call_llm(prompt)
                return ToolResult(output=analysis)

            elif task_mode == "summarize":
                combined_summary = "\n\n".join([f"[{i+1}] {s}" for i, s in enumerate(summaries[:10])])
                return ToolResult(output=f"Retrieved {len(summaries)} docs. Top summaries:\n{combined_summary}")

            else:
                top_k = retrieval_res["results"][:10]
                lines = []
                for i, r in enumerate(top_k, 1):
                    lines.append(f"{i}. {r['fileName']} (score={r['score_hybrid']:.3f})")
                return ToolResult(output=f"Retrieved {len(retrieval_res['results'])} documents. Top 10:\n" + "\n".join(lines))

        except Exception as e:
            return ToolResult(error=f"knowledge_retrieve_literature failed: {str(e)}")
