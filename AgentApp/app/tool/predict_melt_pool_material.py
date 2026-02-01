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

# IMPORTANT: This order must match how you trained your regressors.
# You told me these models were trained on [Velocity, Power, beam D, layer thickness].
FEATURE_ORDER = ["velocity", "power", "beamDiameter", "layerThickness"]  # V, P, D, T

# Map canonical material names (our internal keys) to model *bases* for each target.
# We'll try "<base>.joblib" first (since we're going joblib-only now),
# and fall back to "<base>.json" if you happen to have an XGBoost JSON dump.
REG_MODELS: Dict[str, Dict[str, str]] = {
    "ti-6al-4v": {
        "dep": "./ml_models/Reg/bestMAE_Ti6Al4V_NN_dep_V_P_D_T",
        "len": "./ml_models/Reg/bestMAE_Ti6Al4V_XGB_len_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_Ti6Al4V_NN_wid_V_P_D_T",
    },
    "ss316l": {
        "dep": "./ml_models/Reg/bestMAE_SS316L_GB_dep_V_P_D_T",
        "len": "./ml_models/Reg/bestMAE_SS316L_GP_len_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_SS316L_XGB_wid_V_P_D_T",
    },
    "ss17-4ph": {
        "dep": "./ml_models/Reg/bestMAE_SS174PH_GP_dep_V_P_D_T",
        "len": "./ml_models/Reg/bestMAE_SS174PH_Bootstrap_len_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_SS174PH_GP_wid_V_P_D_T",
    },
    "in718": {
        "dep": "./ml_models/Reg/bestMAE_IN718_GP_dep_V_P_D_T",
        "len": "./ml_models/Reg/bestMAE_IN718_RR_len_V_P_D_T",
        "wid": "./ml_models/Reg/bestMAE_IN718_Bootstrap_wid_V_P_D_T",
    },
}

# Material aliases → our canonical internal keys.
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

    # direct hit?
    if key in REG_MODELS:
        return key

    # alias hit?
    for canon, aliases in MATERIAL_SYNONYMS.items():
        if key == canon or key in aliases:
            return canon

    # substring / loose contains (fallback)
    for canon, aliases in MATERIAL_SYNONYMS.items():
        if any(alias in key for alias in aliases):
            return canon

    return None


def _is_nan(x: Optional[float]) -> bool:
    return (
        x is None
        or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    )


# ----------------------------
# Model file resolution and loading (joblib-first)
# ----------------------------

def _find_model_file(base_no_ext: str) -> Tuple[Optional[str], str]:
    """
    Try to resolve the model file for a given base name.
    Priority:
      1. <base>.joblib   -> kind="joblib"
      2. <base>.json     -> kind="xgb_json" (XGBoost Booster JSON)
    If neither exists: return (None, "missing")
    """
    joblib_path = base_no_ext + ".joblib"
    if os.path.exists(joblib_path):
        return joblib_path, "joblib"

    json_path = base_no_ext + ".json"
    if os.path.exists(json_path):
        return json_path, "xgb_json"

    # last resort: glob in case of minor naming differences
    for pattern, kind in (("*.joblib", "joblib"), ("*.json", "xgb_json")):
        hits = glob.glob(base_no_ext + pattern[1:])
        if hits:
            return hits[0], kind

    return None, "missing"


def _load_model_sync(path: str, kind: str):
    """
    Blocking loader.
    - kind="joblib": load sklearn-like estimators with joblib.load
    - kind="xgb_json": load XGBoostRegressor saved with Booster.save_model("...json")
    """
    if kind == "joblib":
        import joblib
        return joblib.load(path)

    if kind == "xgb_json":
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model

    raise RuntimeError(f"Unsupported model kind '{kind}'")


async def _load_model(path: str, kind: str):
    """
    Async wrapper: run _load_model_sync in a worker thread so we don't block
    the event loop. This still uses joblib.load directly (no subprocess),
    because you've decided to trust that in your environment.
    """
    return await asyncio.to_thread(_load_model_sync, path, kind)


def _predict_scalar(model, arr_2d: np.ndarray) -> Optional[float]:
    """
    Given a loaded regressor (sklearn-like or XGBRegressor) and
    a 2D array [[vel, power, D, T]], return a single float.
    """
    try:
        y = model.predict(arr_2d)
        if y is None:
            return None
        y = np.ravel(y)
        if y.size == 0:
            return None
        return float(y[0])
    except Exception:
        return None


# ----------------------------
# MCP Tool
# ----------------------------

class PredictMeltPool_Material(BaseTool):
    name: str = "predict_melt_pool_material"
    description: str = (
        "Unified LPBF melt-pool geometry regressor for materials {Ti-6Al-4V, SS316L, SS17-4PH, IN718}. "
        "Inputs use LPBF process parameters in normal AM shop units: "
        "velocity [mm/s], power [W], beamDiameter [µm], layerThickness [µm]. "
        "Output is predicted melt pool depth / width / length in µm."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "wc_material": {
                "type": "string",
                "description": (
                    "Material (e.g., 'Ti-6Al-4V', 'SS316L', 'SS17-4PH', 'IN718'). "
                    "Common aliases like 'Ti6Al4V' or 'IN718' also OK."
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
            # 1. Parse Input
            x_raw = _as_obj(input_process_parameters)
            if not isinstance(x_raw, dict):
                return ToolResult(error="input_process_parameters must be a JSON object.")
            
            # 2. Normalize Material
            mat_key = _norm_material_name(wc_material)
            if not mat_key:
                return ToolResult(error=f"Unsupported material '{wc_material}'. Supported: Ti-6Al-4V, SS316L, SS17-4PH, IN718.")

            # 3. Construct Payload
            payload = {
                "material": mat_key,
                "velocity": float(x_raw.get("velocity", 0.0) or 0.0),
                "power": float(x_raw.get("power", 0.0) or 0.0),
                "beamDiameter": float(x_raw.get("beamDiameter", 0.0) or 0.0),
                "layerThickness": float(x_raw.get("layerThickness", 0.0) or 0.0),
            }

            # 4. Call ML Service
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

            # 5. Return Result
            return ToolResult(output=json.dumps(data, ensure_ascii=False))

        except Exception as e:
            return ToolResult(error=f"predict_melt_pool_material failed: {str(e)}")
