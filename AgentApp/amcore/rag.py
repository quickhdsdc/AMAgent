import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

from amcore.labels import canonicalize_material


KB_V2_FILENAMES: Dict[str, str] = {
    "SS316L":      "SS316L",
    "Ti-6Al-4V":   "Ti-6Al-4V",
    "IN718":       "IN718",
    "IN625":       "IN625",
    "17-4PH":      "SS17-4PH",
    "SS17-4PH":    "SS17-4PH",
    "AlSi10Mg":    "AlSi10Mg",
    "18Ni300":     "18Ni300",
    "Hastelloy X": "Hastelloy_X",
    "CuCrZr":      "CuCrZr",
}

LEGACY_FOLDERS: Dict[str, str] = {
    "SS316L":    "SS316L_Search",
    "Ti-6Al-4V": "Ti-6Al-4V_Search",
    "IN718":     "IN718_Search",
    "17-4PH":    "SS17-4PH_Search",
    "SS17-4PH":  "SS17-4PH_Search",
}


class RAGLoader:
    """Material-tagged retrieval with info-score-weighted sampling.

    Returns chunks of paper text (markdown-formatted, ~1500 chars each).
    The KS prompt is built around `Retrieved Literature Evidence:` numbered
    list, so each element returned here becomes one `[1] ...` entry.
    """

    def __init__(self,
                 kb_dir: str = "./bibliography/kb_v2",
                 legacy_dir: str = "./results_AM",
                 min_info_score: float = 0.0,
                 score_temperature: float = 0.8):
        """
        Args:
            kb_dir: path to the per-material JSON files (KB v2).
            legacy_dir: fallback path to the old `results_AM/` corpus.
            min_info_score: drop chunks below this score at load time.
            score_temperature: softer temperatures (>1) flatten weighting
                toward uniform; smaller (<1) sharpen toward the richest
                chunks. 0.8 is a mild preference for higher-score chunks.
        """
        self.kb_dir = kb_dir
        self.legacy_dir = legacy_dir
        self.min_info_score = min_info_score
        self.score_temperature = score_temperature
        self._cache: Dict[str, List[Tuple[str, float]]] = {}

    def _load_kb_v2(self, canonical: str) -> Optional[List[Tuple[str, float]]]:
        stem = KB_V2_FILENAMES.get(canonical)
        if stem is None:
            return None
        path = os.path.join(self.kb_dir, f"{stem}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"[RAG] failed to load {path}: {e}")
            return None
        out: List[Tuple[str, float]] = []
        for c in chunks:
            text = (c.get("content") or "").strip()
            if not text:
                continue
            score = float(c.get("info_score", 0.0))
            if score < self.min_info_score:
                continue
            out.append((text, score))
        return out if out else None

    def _load_legacy(self, canonical: str) -> List[Tuple[str, float]]:
        """Read the old results_AM/<Mat>_Search/results.jsonl summaries.

        Returns each summary with a uniform score so weighted sampling
        degrades gracefully to the previous behavior.
        """
        folder = LEGACY_FOLDERS.get(canonical)
        if not folder:
            return []
        path = os.path.join(self.legacy_dir, folder, "results.jsonl")
        if not os.path.exists(path):
            return []
        out: List[Tuple[str, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                summ = (rec.get("summary") or "").strip()
                if len(summ) < 50:
                    continue
                if "No relevant parameter-defect information" in summ:
                    continue
                out.append((summ, 1.0))
        if out:
            print(f"[RAG] (legacy) {len(out)} summaries for {canonical}")
        return out

    def load_material_corpus(self, canonical: str) -> List[Tuple[str, float]]:
        if canonical in self._cache:
            return self._cache[canonical]

        items = self._load_kb_v2(canonical)
        source = "kb_v2"
        if items is None:
            items = self._load_legacy(canonical)
            source = "legacy"

        if not items:
            print(f"[RAG] no corpus for {canonical}")
            self._cache[canonical] = []
            return []

        self._cache[canonical] = items
        print(f"[RAG] loaded {len(items)} chunks for {canonical} (source={source})")
        return items

    def get_context(self, material: str, k: int = 15,
                    seed: int = 42,
                    process_params: Optional[Dict[str, float]] = None,
                    alpha: float = 1.5,
                    beta: float = 1.0) -> List[str]:
        """Return `k` chunk texts for the given material.

        If `process_params` is None: legacy info-score weighted random
        sampling using `seed`.

        If `process_params` is provided: deterministic top-k by combined
        relevance = info_score + α·numeric_overlap + β·has_ved_and_defect.
        `seed` is ignored in this mode.
        """
        canonical = canonicalize_material(material)
        if not canonical:
            return []
        pool = self.load_material_corpus(canonical)
        if not pool:
            return []
        k_actual = min(k, len(pool))

        if process_params:
            scored = []
            for text, info_score in pool:
                ov = _numeric_overlap_score(text, process_params)
                vd = 1.0 if _has_ved_and_defect(text) else 0.0
                rel = info_score + alpha * ov + beta * vd
                scored.append((rel, text))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:k_actual]]

        rng = random.Random(seed)
        scores = [s for _, s in pool]
        min_s = min(scores)
        shifted = [s - min_s + 0.5 for s in scores]
        if self.score_temperature != 1.0:
            shifted = [w ** (1.0 / self.score_temperature) for w in shifted]
        chosen_idx: List[int] = []
        avail = list(range(len(pool)))
        for _ in range(k_actual):
            if not avail:
                break
            i = rng.choices(range(len(avail)),
                            weights=[shifted[j] for j in avail], k=1)[0]
            chosen_idx.append(avail.pop(i))
        return [pool[i][0] for i in chosen_idx]


_PARAM_UNITS: Dict[str, Tuple[Tuple[str, ...], Tuple[float, float]]] = {
    "Power":           (("w",),                  (10, 3000)),
    "Velocity":        (("mm/s", "m/s"),          (10, 8000)),
    "beam D":          (("µm", "um", "mm"),       (10, 500)),
    "layer thickness": (("µm", "um", "mm"),       (5, 200)),
    "Hatch spacing":   (("µm", "um", "mm"),       (10, 400)),
}

_NUM_UNIT_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>W\b|kW\b|mm\s*/\s*s|m\s*/\s*s|µm|um|mm(?!\s*/)|J\s*/?\s*mm[³^3]?|%)",
    re.IGNORECASE,
)

_VED_DEFECT_RE = re.compile(
    r"(VED|J\s*/?\s*mm[³^3]|volumetric\s+energy\s+density).{0,120}"
    r"(keyhole|lack[- ]of[- ]fusion|\blof\b|balling|porosity)",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_value_to_param_unit(num: float, unit: str,
                                  target_units: Tuple[str, ...]) -> Optional[float]:
    """Convert a (num, unit) pair to the param's natural unit.

    e.g., target_units=('mm/s', 'm/s'): an m/s reading gets *1000.
    target_units=('µm','um','mm'): an mm reading gets *1000.
    """
    u = unit.lower().replace(" ", "")
    if any(tu in u for tu in target_units if tu != "mm"):
        return num
    if "m/s" in u and "mm/s" in target_units:
        return num * 1000.0
    if "kw" in u and "w" in target_units:
        return num * 1000.0
    if "w" in u and "w" in target_units:
        return num
    if u == "mm" and any(tu in ("µm", "um") for tu in target_units):
        return num * 1000.0
    return None


def _numeric_overlap_score(chunk_text: str,
                          query_params: Dict[str, float]) -> float:
    """Count (number, unit) tokens in the chunk that match each query
    parameter's unit family AND have similar magnitude.

    Full credit (1.0) if within ±30%; partial (0.3) if within ±100%.
    Returns the sum across all query params. Capped at 6 to prevent a
    single ultra-dense chunk from dominating.
    """
    pairs = []
    for m in _NUM_UNIT_RE.finditer(chunk_text):
        try:
            num = float(m.group("num"))
        except ValueError:
            continue
        pairs.append((num, m.group("unit")))
    if not pairs:
        return 0.0

    total = 0.0
    for param, qval in query_params.items():
        if qval is None:
            continue
        try:
            qval = float(qval)
        except (TypeError, ValueError):
            continue
        if qval <= 0:
            continue
        spec = _PARAM_UNITS.get(param)
        if spec is None:
            continue
        target_units, (mn, mx) = spec
        best = 0.0
        for num, unit in pairs:
            n_param = _normalize_value_to_param_unit(num, unit, target_units)
            if n_param is None:
                continue
            if not (mn <= n_param <= mx):
                continue
            ratio = max(n_param, qval) / max(min(n_param, qval), 1e-6)
            if ratio <= 1.3:
                best = max(best, 1.0)
            elif ratio <= 2.0:
                best = max(best, 0.3)
        total += best
    return min(total, 6.0)


def _has_ved_and_defect(chunk_text: str) -> bool:
    return bool(_VED_DEFECT_RE.search(chunk_text))


_RAG_LOADER: Optional[RAGLoader] = None


def get_rag_loader() -> RAGLoader:
    global _RAG_LOADER
    if _RAG_LOADER is None:
        _RAG_LOADER = RAGLoader()
    return _RAG_LOADER


def reset_rag_loader() -> None:
    """Used by tests / benchmarks that want to switch corpus on the fly."""
    global _RAG_LOADER
    _RAG_LOADER = None
