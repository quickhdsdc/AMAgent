import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_AMAGENT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _AMAGENT_ROOT not in sys.path:
    sys.path.insert(0, _AMAGENT_ROOT)

from app.tool.base import BaseTool, ToolResult

from amcore.labels import canonicalize_material
from amcore.rag_v2 import retrieve_evidence_rows, get_loader as get_rag_v2_loader
from amcore.evidence_pack import build_evidence_pack
from amcore.llm_client import call_llm


_PARAM_ALIASES = {
    "power":           "Power",
    "p":               "Power",
    "Power":           "Power",
    "velocity":        "Velocity",
    "scan_velocity":   "Velocity",
    "scanVelocity":    "Velocity",
    "v":               "Velocity",
    "Velocity":        "Velocity",
    "beamD":           "beam D",
    "beam_d":          "beam D",
    "beamDiameter":    "beam D",
    "beam_diameter":   "beam D",
    "beam D":          "beam D",
    "layerThickness":  "layer thickness",
    "layer_thickness": "layer thickness",
    "layer thickness": "layer thickness",
    "hatchSpacing":    "Hatch spacing",
    "hatch_spacing":   "Hatch spacing",
    "Hatch spacing":   "Hatch spacing",
    "material":        "material",
    "wc_material":     "material",
}


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json(s: str) -> Any:
    s = _strip_fences(s)
    return json.loads(s)


def _coerce_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(x))
        return float(m.group(0)) if m else None


def _normalize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map alias keys to canonical keys and coerce numeric values."""
    out: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
        canon = _PARAM_ALIASES.get(k, k)
        out[canon] = v
    for k in ("Power", "Velocity", "beam D", "layer thickness", "Hatch spacing"):
        if k in out:
            out[k] = _coerce_float(out[k])
    return out


def _ood_fallback(name: str) -> str:
    """Heuristic mapping for materials not in the alias table."""
    s = (name or "").lower().strip()
    if "ti" in s or "titanium" in s:
        return "Ti-6Al-4V"
    if "in718" in s or "inconel" in s or "ni" in s or "alloy" in s:
        return "IN718"
    if "alsi" in s or "aluminum" in s or "aluminium" in s:
        return "AlSi10Mg"
    if "hastelloy" in s:
        return "Hastelloy X"
    if "marag" in s or "18ni" in s:
        return "18Ni300"
    if "17-4" in s or "17 4" in s:
        return "17-4PH"
    return "SS316L"


def _resolve_material(raw_name: str) -> Dict[str, Any]:
    canon = canonicalize_material(raw_name)
    if canon:
        return {"canonical": canon, "is_ood": False, "fallback_from": None}
    fb = _ood_fallback(raw_name)
    return {"canonical": fb, "is_ood": True, "fallback_from": raw_name}


def _val_str(params: Dict[str, Any], key: str, unit: str) -> str:
    v = params.get(key)
    return f"{v} {unit}" if v is not None else f"unknown {unit}"


def _build_kd_prompt(params: Dict[str, Any],
                    evidence_pack: str,
                    is_ood: bool,
                    original_material: str,
                    search_material: str) -> str:
    material = params.get("material", "unknown")
    prompt = (
        "You are a Knowledge-Driven LPBF process analysis assistant. Assess "
        f"the potential imperfections for {material} manufactured at "
        f"{_val_str(params, 'Power', 'W')}, beam {_val_str(params, 'beam D', 'µm')}, "
        f"velocity {_val_str(params, 'Velocity', 'mm/s')}, layer "
        f"{_val_str(params, 'layer thickness', 'µm')}, hatch "
        f"{_val_str(params, 'Hatch spacing', 'µm')}.\n\n"
        "Retrieved Literature Evidence:\n"
        f"{evidence_pack}\n\n"
        "Task:\n"
        "1. Treat the table above as a nearest-neighbor parameter table — each "
        "row is a prior (P, v, h, t, VED) → (defect, density) datapoint on the "
        "same material. Rows with smaller VED gap to the query are stronger "
        "evidence. A direct density measurement at near-identical parameters "
        "dominates window-only reasoning.\n"
        "2. Process-window check: if parameters fall inside the cited optimal "
        "window AND no row reports a defect at similar (P, v, h, t), default "
        "to 'none'.\n"
        "3. Reliability:\n"
        "   - HIGH (0.7-1.0) if a direct numerical match drives the call.\n"
        "   - MEDIUM (0.3-0.7) if window reasoning grounded in this material.\n"
        "   - LOW (0.1-0.3) if neither — fall back to internal physics.\n"
    )
    if is_ood:
        prompt += (
            f"   - OOD penalty: '{original_material}' is out of distribution. "
            f"Evidence is drawn from '{search_material}'. Multiply your "
            "reliability by 0.5 and state this explicitly in [THINK].\n"
        )
    prompt += (
        "4. Assumptions: many papers focus on failures; do not assume a defect "
        "exists if the nearest rows report nominal/standard outcomes.\n"
        "5. Estimate belief distribution over {none, lof, balling, keyhole}. "
        "Keep each component ≥ 0.05.\n"
        "6. Conclude with a single [LABEL].\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {nearest-neighbor reading + window check} [/THINK]\n"
        "[ASSUMPTIONS] {assumptions and numeric mismatch warnings} "
        "[/ASSUMPTIONS]\n"
        "[RELIABILITY] {0.0 to 1.0} [/RELIABILITY]\n"
        "[BELIEF] {\"none\": 0.X, \"lof\": 0.X, \"balling\": 0.X, "
        "\"keyhole\": 0.X} [/BELIEF]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt


class KnowledgeRetrievalLiterature(BaseTool):
    name: str = "knowledge_retrieve_literature"
    description: str = (
        "Retrieve structured LPBF evidence rows for the queried process "
        "parameters and run the Knowledge-Driven Analyst. Returns a "
        "schema-tagged belief over defect labels grounded in nearest-"
        "parameter evidence from the literature KB."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": (
                    "JSON object with process_terms, material_terms, "
                    "parameter_terms, objective_terms. Currently used only "
                    "for logging; retrieval is driven by the numerical "
                    "parameters."
                ),
            },
            "input_process_parameters": {
                "type": "string",
                "description": (
                    "JSON object of process parameters: material, Power, "
                    "Velocity, beam D, layer thickness, Hatch spacing."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Number of evidence rows to retrieve (default 6).",
            },
        },
        "required": ["keywords", "input_process_parameters"],
    }

    async def execute(self, keywords: str, input_process_parameters: str,
                     k: int = 6, **kwargs) -> ToolResult:
        try:
            params_raw = _parse_json(input_process_parameters)
            if not isinstance(params_raw, dict):
                return ToolResult(error="input_process_parameters must be a JSON object")

            params = _normalize_params(params_raw)
            raw_material = str(params.get("material") or "").strip()
            resolved = _resolve_material(raw_material)
            search_material = resolved["canonical"]
            is_ood = resolved["is_ood"]
            params["material"] = search_material

            print(f"[knowledge_tool] query material='{raw_material}' -> "
                 f"canonical='{search_material}' (OOD={is_ood})")

            ev_rows = retrieve_evidence_rows(
                {"material": search_material,
                 "Power":            params.get("Power"),
                 "Velocity":         params.get("Velocity"),
                 "Hatch spacing":    params.get("Hatch spacing"),
                 "layer thickness":  params.get("layer thickness"),
                 "beam D":           params.get("beam D")},
                k=int(k),
                loader=get_rag_v2_loader(),
            )

            evidence_pack = build_evidence_pack(
                ev_rows,
                {"material": search_material, **params},
            )

            prompt = _build_kd_prompt(
                params,
                evidence_pack=evidence_pack,
                is_ood=is_ood,
                original_material=raw_material,
                search_material=search_material,
            )

            llm_response = call_llm(prompt, profile="default")

            audit = {
                "raw_material":   raw_material,
                "search_material": search_material,
                "is_ood":         is_ood,
                "n_evidence_rows": len(ev_rows),
                "evidence_rows":  [
                    {"rank": r.get("rank"), "P_W": r.get("P_W"),
                     "v_mms": r.get("v_mms"), "VED": r.get("VED_Jmm3"),
                     "defect": r.get("defect"), "density": r.get("density_pct"),
                     "source": r.get("source_file"),
                     "rel": r.get("audit", {}).get("rel")}
                    for r in ev_rows
                ],
            }
            payload = {
                "kd_response": llm_response,
                "retrieval_audit": audit,
            }
            return ToolResult(output=json.dumps(payload, ensure_ascii=False))

        except Exception as e:
            return ToolResult(error=f"knowledge_retrieve_literature failed: {e}")
