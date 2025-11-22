from app.tool.base import BaseTool, ToolResult
from typing import Any, Dict, Optional, Tuple, List, Union
import json
import re
import math

# ----------------------------
# Helpers: safe parsing
# ----------------------------
def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _as_obj(text: Union[str, Dict[str, Any]]) -> Any:
    """
    Accept a dict directly or find the first complete JSON value in a string.
    Handles objects `{...}` or arrays `[...]`, ignores junk around it,
    and copes with code fences. Raises ValueError if none found.
    """
    if isinstance(text, dict):
        return text
    s = _strip_fences(str(text))
    decoder = json.JSONDecoder()
    i, n = 0, len(s)
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

def _num(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None

def _norm_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    return u.strip()

def _sig6(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        return float(f"{x:.6g}")
    except Exception:
        return x

# ----------------------------
# Canonicalization maps
# ----------------------------
_PROCESS_SYNONYMS = {
    "lpbf": "LPBF",
    "laser powder bed fusion": "LPBF",
    "laser-powder-bed-fusion": "LPBF",
    "slm": "SLM",
    "selective laser melting": "SLM",
}
_ALLOWED_PROCESSES = {"LPBF", "SLM"}

_MAT_SYNONYMS = {
    "in625": "IN625", "inconel 625": "IN625", "alloy 625": "IN625",
    "in718": "IN718", "inconel 718": "IN718", "alloy 718": "IN718",
    "17-4ph": "SS17-4PH", "17-4 ph": "SS17-4PH", "ss17-4ph": "SS17-4PH",
    "stainless 17-4": "SS17-4PH", "17-4 stainless": "SS17-4PH",
    "316l": "SS316L", "ss316l": "SS316L", "stainless 316l": "SS316L",
    "ti-6al-4v": "Ti-6Al-4V", "ti6al4v": "Ti-6Al-4V", "ti64": "Ti-6Al-4V",
}
_ALLOWED_MATERIALS = {"IN625", "IN718", "SS17-4PH", "SS316L", "Ti-6Al-4V"}

# ----------------------------
# Unit conversion helpers
# (Targets: engineering units, not SI)
# ----------------------------
def _convert_power_to_W(val: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], bool]:
    if val is None:
        return None, False
    u = (_norm_unit(unit) or "").lower()
    if u in ("w", "watt", "watts", ""):
        return val, False
    if u in ("kw", "kilowatt", "kilowatts"):
        return val * 1000.0, False
    if unit is not None:
        return None, True
    return None, False

def _vel_to_mmps(val: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], bool]:
    """
    Convert velocity-like value to mm/s.
    Returns (value_in_mm/s, unit_unrecognized_flag).
    """
    if val is None:
        return None, False
    u = (_norm_unit(unit) or "").lower()
    if u in ("mm/s", "mmps", ""):
        return val, False
    if u in ("m/s", "mps"):
        return val * 1e3, False
    if u in ("cm/s", "cmps"):
        return val * 10.0, False
    if unit is not None:
        return None, True
    return None, False

def _len_to_um(val: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], bool]:
    """
    Convert length-like value to micrometers (µm).
    Returns (value_in_µm, unit_unrecognized_flag).
    """
    if val is None:
        return None, False
    u = (_norm_unit(unit) or "").lower()
    if u in ("µm", "μm", "um", "micrometer", "micrometers", "micron", "microns", ""):
        return val, False
    if u in ("mm", "millimeter", "millimeters"):
        return val * 1e3, False
    if u in ("m", "meter", "meters"):
        return val * 1e6, False
    if u in ("nm", "nanometer", "nanometers"):
        return val * 1e-3, False
    if unit is not None:
        return None, True
    return None, False

# Material properties units (SI)
def _clean_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip().lower()
    s = s.replace("⋅", "·").replace("·", "*").replace("-", "*").replace(" ", "")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"\*{2,}", "*", s)
    s = re.sub(r"/{2,}", "/", s)
    return s

def _cp_to_j_per_kgk(val: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], bool]:
    if val is None:
        return None, False
    u = _clean_unit(unit)
    if u in (None, ""):
        return None, False
    is_kilo = False
    if u.startswith("kj"):
        is_kilo = True
        u = "j" + u[2:]
    if re.fullmatch(r"j/kg([*/]|/)k", u) or re.fullmatch(r"j/kgk", u):
        return (val * 1000.0 if is_kilo else val), False
    return None, True

def _k_to_w_per_mk(val: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], bool]:
    if val is None:
        return None, False
    u = _clean_unit(unit)
    if u in (None, ""):
        return None, False
    if re.fullmatch(r"w/m([*/]|/)k", u) or re.fullmatch(r"w/mk", u):
        return val, False
    return None, True

# ----------------------------
# Ranges (engineering units)
# ----------------------------
_REF_RANGES = {
    "laser_power":     {"min": 50.0,  "max": 500.0, "unit": "W"},
    "scan_velocity":   {"min": 0.0,  "max": 7000.0, "unit": "mm/s"},
    "beam_diameter":   {"min": 50.0,  "max": 100.0,  "unit": "µm"},
    "layer_thickness": {"min": 20.0,  "max": 100.0,  "unit": "µm"},
    "hatch_spacing":   {"min": 80.0,  "max": 120.0,  "unit": "µm"},
}

def _outside_with_margin(val: float, key: str) -> bool:
    r = _REF_RANGES[key]
    lo, hi = r["min"], r["max"]
    if val < 0:
        return True
    if val < lo and (lo - val) / max(lo, 1e-12) > 0.5:
        return True
    if val > hi and (val - hi) / max(hi, 1e-12) > 0.5:
        return True
    return False

# ----------------------------
# Canonicalization helpers
# ----------------------------
def _canon_process(raw: Optional[str]) -> Tuple[Optional[str], bool, List[str]]:
    if raw is None:
        return None, True, []
    s = raw.strip().lower()
    aliases: List[str] = []
    if s in _PROCESS_SYNONYMS:
        canon = _PROCESS_SYNONYMS[s]
        aliases.append(raw)
    else:
        for k, v in _PROCESS_SYNONYMS.items():
            if k in s:
                canon = v
                aliases.append(raw)
                break
        else:
            return raw, True, []
    return canon, (canon not in _ALLOWED_PROCESSES), aliases

def _canon_material(raw: Optional[str]) -> Tuple[Optional[str], bool, List[str]]:
    if raw is None:
        return None, True, []
    s = raw.strip().lower()
    aliases: List[str] = []
    if s in _MAT_SYNONYMS:
        canon = _MAT_SYNONYMS[s]
        aliases.append(raw)
    else:
        for k, v in _MAT_SYNONYMS.items():
            if k in s:
                canon = v
                aliases.append(raw)
                break
        else:
            return raw, True, []
    return canon, (canon not in _ALLOWED_MATERIALS), aliases

def _canon_param_block(ppr: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str], bool, List[str]]:
    """
    Returns:
      (process_parameters, unit_unrec_flag, unit_offenders, ood_param_flag, ood_offenders)
    All targets in ENGINEERING units:
      power[W], velocity[mm/s], beam diameter[µm], layer thickness[µm], hatch spacing[µm].
    """
    unit_unrec = False
    unit_offenders: List[str] = []
    ood = False
    ood_off: List[str] = []
    out: Dict[str, Dict[str, Optional[float]]] = {}

    # laser_power → W
    v = _num((ppr.get("laser_power") or {}).get("value"))
    u = (ppr.get("laser_power") or {}).get("unit")
    v2, unrec = _convert_power_to_W(v, u)
    if unrec: unit_unrec, unit_offenders = True, unit_offenders + ["laser_power"]
    out["laser_power"] = {"value": _sig6(v2), "unit": "W" if v2 is not None else None}
    if v2 is not None and _outside_with_margin(v2, "laser_power"):
        ood, ood_off = True, ood_off + ["laser_power"]

    # scan_velocity → mm/s
    v = _num((ppr.get("scan_velocity") or {}).get("value"))
    u = (ppr.get("scan_velocity") or {}).get("unit")
    v2, unrec = _vel_to_mmps(v, u)
    if unrec: unit_unrec, unit_offenders = True, unit_offenders + ["scan_velocity"]
    out["scan_velocity"] = {"value": _sig6(v2), "unit": "mm/s" if v2 is not None else None}
    if v2 is not None and _outside_with_margin(v2, "scan_velocity"):
        ood, ood_off = True, ood_off + ["scan_velocity"]

    # beam_diameter → µm
    v = _num((ppr.get("beam_diameter") or {}).get("value"))
    u = (ppr.get("beam_diameter") or {}).get("unit")
    v2, unrec = _len_to_um(v, u)
    if unrec: unit_unrec, unit_offenders = True, unit_offenders + ["beam_diameter"]
    out["beam_diameter"] = {"value": _sig6(v2), "unit": "µm" if v2 is not None else None}
    if v2 is not None and _outside_with_margin(v2, "beam_diameter"):
        ood, ood_off = True, ood_off + ["beam_diameter"]

    # layer_thickness → µm
    v = _num((ppr.get("layer_thickness") or {}).get("value"))
    u = (ppr.get("layer_thickness") or {}).get("unit")
    v2, unrec = _len_to_um(v, u)
    if unrec: unit_unrec, unit_offenders = True, unit_offenders + ["layer_thickness"]
    out["layer_thickness"] = {"value": _sig6(v2), "unit": "µm" if v2 is not None else None}
    if v2 is not None and _outside_with_margin(v2, "layer_thickness"):
        ood, ood_off = True, ood_off + ["layer_thickness"]

    # hatch_spacing → µm
    v = _num((ppr.get("hatch_spacing") or {}).get("value"))
    u = (ppr.get("hatch_spacing") or {}).get("unit")
    v2, unrec = _len_to_um(v, u)
    if unrec: unit_unrec, unit_offenders = True, unit_offenders + ["hatch_spacing"]
    out["hatch_spacing"] = {"value": _sig6(v2), "unit": "µm" if v2 is not None else None}
    if v2 is not None and _outside_with_margin(v2, "hatch_spacing"):
        ood, ood_off = True, ood_off + ["hatch_spacing"]

    return out, unit_unrec, unit_offenders, ood, ood_off

def _canon_material_props(props: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
    unit_unrec = False
    offenders: List[str] = []

    # Cp
    v = _num((props.get("Cp") or {}).get("value"))
    u = (props.get("Cp") or {}).get("unit")
    v2, unrec = _cp_to_j_per_kgk(v, u)
    if unrec: unit_unrec, offenders = True, offenders + ["material.properties.Cp"]
    Cp = {"value": _sig6(v2), "unit": "J/(kg·K)" if v2 is not None else None}

    # k
    v = _num((props.get("k") or {}).get("value"))
    u = (props.get("k") or {}).get("unit")
    v2, unrec = _k_to_w_per_mk(v, u)
    if unrec: unit_unrec, offenders = True, offenders + ["material.properties.k"]
    k = {"value": _sig6(v2), "unit": "W/(m·K)" if v2 is not None else None}

    return {"Cp": Cp, "k": k}, unit_unrec, offenders

# ----------------------------
# MCP Tool
# ----------------------------
class TaskCanonicalizeParam(BaseTool):
    name: str = "task_canonicalize_param"
    description: str = (
        "Canonicalize and validate an AM query struct from task_partition_param: "
        "convert units to engineering targets (W, mm/s, µm), normalize process/material names, "
        "and emit validity flags & ranges."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "payload_json": {
                "type": "string",
                "description": "JSON string (or object) from task_partition_param containing working_conditions, process_parameters_raw, and prediction_objective.",
            }
        },
        "required": ["payload_json"]
    }

    async def execute(self, payload_json: Union[str, Dict[str, Any]], **kwargs) -> ToolResult:
        try:
            data = _as_obj(payload_json)

            wc_in = data.get("working_conditions") or {}
            ppr_in = data.get("process_parameters_raw") or {}
            objective = data.get("prediction_objective")

            # Process / Material canonicalization
            raw_proc = wc_in.get("process")
            canon_proc, ood_proc, proc_aliases = _canon_process(raw_proc)

            mat_in = wc_in.get("material") or {}
            raw_mat = mat_in.get("name")
            canon_mat, ood_mat, mat_aliases = _canon_material(raw_mat)

            props_in = (mat_in.get("properties") or {})
            mat_props, unit_unrec_props, unit_off_props = _canon_material_props(props_in)

            # Parameters canonicalization (engineering units)
            params_out, unit_unrec_params, unit_off_params, ood_params, ood_off_params = _canon_param_block(ppr_in)

            # Validity flags & offenders
            unit_unrecognized = unit_unrec_props or unit_unrec_params
            unit_offending = unit_off_props + unit_off_params

            critical_missing_fields: List[str] = []
            for key in ("laser_power", "scan_velocity", "beam_diameter", "layer_thickness", "hatch_spacing"):
                if params_out.get(key, {}).get("value") is None:
                    critical_missing_fields.append(key)
            critical_missing = len(critical_missing_fields) > 0

            flags = {
                "critical_missing": critical_missing,
                "unit_unrecognized": unit_unrecognized,
                "ood_process": ood_proc,
                "ood_material": ood_mat,
                "ood_parameters": ood_params,
            }
            offending_fields = {
                "critical_missing": critical_missing_fields,
                "unit_unrecognized": unit_offending,
                "ood_process": ([] if not ood_proc else [raw_proc] if raw_proc is not None else ["process"]),
                "ood_material": ([] if not ood_mat else [raw_mat] if raw_mat is not None else ["material"]),
                "ood_parameters": ood_off_params,
            }

            validity = {
                "status": "fail" if any(flags.values()) else "pass",
                "flags": flags,
                "offending_fields": offending_fields,
                "ranges_used": {
                    k: {"min": v["min"], "max": v["max"], "unit": v["unit"]}
                    for k, v in _REF_RANGES.items()
                },
            }

            working_conditions = {
                "process": canon_proc if canon_proc is not None else None,
                "material": {
                    "name": canon_mat if canon_mat is not None else None,
                    "aliases": mat_aliases,
                    "properties": mat_props,
                },
            }

            out = {
                "working_conditions": working_conditions,
                "process_parameters": params_out,
                "validity": validity,
                "prediction_objective": objective if objective in ("regression", "classification", "both") else "both",
            }

            return ToolResult(output=json.dumps(out, ensure_ascii=False))

        except Exception as e:
            return ToolResult(error=f"task_canonicalize_param failed: {str(e)}")





