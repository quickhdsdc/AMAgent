import os, json, math, re, glob, asyncio
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
from app.tool.base import BaseTool, ToolResult


# ----------------------------
# Helpers: parse possibly-messy JSON input
# ----------------------------
def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _as_obj(text: Union[str, Dict[str, Any]]) -> Any:
    """
    Accepts:
      - dict (returned as-is)
      - string that MAY contain leading/trailing junk or code fences
    We scan for first JSON object/array and decode it.
    """
    if isinstance(text, dict):
        return text

    s = _strip_fences(str(text))
    dec = json.JSONDecoder()
    i, n = 0, len(s)

    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] in "{[":
            try:
                obj, end = dec.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
                i += 1
                continue
        i += 1

    raise ValueError("No complete JSON object/array found in input.")

# ----------------------------
# Domain config
# ----------------------------
REG_MODELS: Dict[str, Dict[str, str]] = {
    "ti-6al-4v": {
        "len": "./ml_models/Reg/bestMAE_Ti6Al4V_XGB_len_H_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_Ti6Al4V_XGB_wid_H_V_P_D_T",
        "dep": "./ml_models/Reg/bestMAE_Ti6Al4V_XGB_dep_H_V_P_D_T",
    },
    "ss316l": {
        "len": "./ml_models/Reg/bestMAE_SS316L_XGB_len_H_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_SS316L_NN_wid_H_V_P_D_T",
        "dep": "./ml_models/Reg/bestMAE_SS316L_GP_dep_H_V_P_D_T",        
    },
    "ss17-4ph": {
        "len": "./ml_models/Reg/bestMAE_SS174PH_XGB_len_H_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_SS174PH_GP_wid_H_V_P_D_T",
        "dep": "./ml_models/Reg/bestMAE_SS174PH_GP_dep_H_V_P_D_T",        
    },
    "in718": {
        "wid": "./ml_models/Reg/bestMAE_IN718_GP_wid_H_V_P_D_T",
        "dep": "./ml_models/Reg/bestMAE_IN718_NN_dep_H_V_P_D_T",        
    },
}

MATERIAL_SYNONYMS: Dict[str, List[str]] = {
    "ti-6al-4v": ["ti-6al-4v", "ti6al4v", "ti 6al 4v", "ti64", "grade 5"],
    "ss316l":    ["ss316l", "316l", "aisi 316l"],
    "ss17-4ph":  ["ss17-4ph", "17-4ph", "17-4 ph", "17-4 stainless", "17-4 precipitation hardening"],
    "in718":     ["in718", "inconel 718", "alloy 718"],
}

def _norm_material_name(s: str) -> Optional[str]:
    """
    Normalize wc_material (user may say 'Ti6Al4V', 'IN718', etc.)
    Return our canonical internal key or None if unsupported.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    key = s.strip().lower()

    if key in REG_MODELS:
        return key

    for canon, aliases in MATERIAL_SYNONYMS.items():
        if key == canon or key in aliases:
            return canon

    for canon, aliases in MATERIAL_SYNONYMS.items():
        if any(alias in key for alias in aliases):
            return canon

    return None

def _predict_similar_material(name: str) -> str:
    """
    If material is OOD, map to closest known based on simple heuristics.
    defult fallback: ss316l.
    """
    s = str(name).lower().strip()
    if "ti" in s or "titanium" in s:
        return "ti-6al-4v"
    if "in718" in s or "inconel" in s or "nickel" in s or "alloy" in s:
        return "in718"
    if "ss" in s or "steel" in s or "iron" in s or "fe" in s:
        if "17-4" in s or "ph" in s:
            return "ss17-4ph"
        return "ss316l"
    return "ss316l"



def _calculate_reliability(is_ood_mat: bool, is_ood_param: bool) -> float:
    score = 0.9 
    if is_ood_mat:
        score *= 0.5
    if is_ood_param:
        score *= 0.7
    return float(round(score, 2))



def _is_nan(x: Optional[float]) -> bool:
    return (
        x is None
        or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    )


# ----------------------------
# MCP Tool
# ----------------------------

class PredictMeltPool_Material(BaseTool):
    name: str = "predict_melt_pool_material"
    description: str = (
        "Unified LPBF melt-pool geometry regressor"
        "Inputs use LPBF process parameters in normal AM shop units: "
        "velocity [mm/s], power [W], beamDiameter [µm], layerThickness [µm], hatchSpacing [µm]. "
        "Output is predicted melt pool depth / width / length in µm."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "wc_material": {
                "type": "string",
                "description": (
                    "Material name (e.g., 'Ti-6Al-4V', 'SS316L', 'SS17-4PH', 'IN718'). "
                ),
            },
            "input_process_parameters": {
                "type": "object",
                "description": (
                    "Dict or JSON string with process params in standard AM units: "
                    "{'velocity': <mm/s>, 'power': <W>, 'beamDiameter': <µm>, 'layerThickness': <µm>, 'hatchSpacing': <µm>} "
                ),
            },
        },
        "required": ["wc_material", "input_process_parameters"],
        "additionalProperties": False,
    }

    async def execute(self, wc_material: str, input_process_parameters: Union[str, Dict[str, Any]], **kwargs) -> ToolResult:
        try:
            x_raw = _as_obj(input_process_parameters)
            if not isinstance(x_raw, dict):
                return ToolResult(error="input_process_parameters must be a JSON object.")
            
            mat_key = _norm_material_name(wc_material)
            
            is_ood_mat = False
            if not mat_key:
                is_ood_mat = True
                mat_key = _predict_similar_material(wc_material)

            payload = {
                "material": mat_key,
                "velocity": float(x_raw.get("velocity", 0.0) or 0.0),
                "power": float(x_raw.get("power", 0.0) or 0.0),
                "beamDiameter": float(x_raw.get("beamDiameter", 0.0) or 0.0),
                "layerThickness": float(x_raw.get("layerThickness", 0.0) or 0.0),
                "hatchSpacing": float(x_raw.get("hatchSpacing", 0.0) or 0.0),
            }

            import httpx
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post("http://localhost:8000/predict/meltpool", json=payload, timeout=5.0)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.RequestError:
                     return ToolResult(error="ML Service unavailable (connection refused). Please ensure app/ml_service.py is running.")
                except httpx.HTTPStatusError as e:
                     return ToolResult(error=f"ML Service error: {e.response.text}")
            
            is_ood_param = False
            
            reliability = _calculate_reliability(is_ood_mat, is_ood_param)
            
            data["material_used"] = mat_key
            data["is_ood_material"] = is_ood_mat
            data["is_ood_params"] = is_ood_param
            data["reliability"] = reliability

            return ToolResult(output=json.dumps(data, ensure_ascii=False))

        except Exception as e:
            return ToolResult(error=f"predict_melt_pool_material failed: {str(e)}")
