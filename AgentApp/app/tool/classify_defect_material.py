from app.tool.base import BaseTool, ToolResult
from typing import Dict, Any, Optional, List, Union
import os, json, math, re, time
import numpy as np


try:
    import joblib
except Exception:
    joblib = None

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _as_obj(text: Union[str, Dict[str, Any]]) -> Any:
    if isinstance(text, dict):
        return text
    s = _strip_fences(str(text))
    decoder = json.JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n: break
        if s[i] in "{[":
            try:
                obj, end = decoder.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
                i += 1
                continue
        i += 1
    raise ValueError("No complete JSON object/array found in input.")

_FEATURE_ORDER = ["velocity", "power", "beamDiameter", "layerThickness"] 

_RANGES_USER_UNITS = {
    "velocity":       (50.0, 3000.0), 
    "power":          (50.0, 1000.0), 
    "beamDiameter":   (10.0, 200.0),  
    "layerThickness": (10.0, 100.0),  
}

LABEL_ORDER = ["none", "LoF", "balling", "keyhole"] 
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_ID.items()}

_MATERIAL_MODELS = {
    "ti-6al-4v": "./ml_models/Cls/bestF1_Ti6Al4V_GP_V_P_D_T.joblib",
    "ss316l":    "./ml_models/Cls/bestF1_SS316L_XGB_V_P_D_T.joblib",
    "ss17-4ph":  "./ml_models/Cls/bestF1_SS174PH_RF_V_P_D_T.joblib",
    "in718":     "./ml_models/Cls/bestF1_IN718_GP_V_P_D_T.joblib",
}

_MATERIAL_SYNONYMS = {
    "ti-6al-4v": ["ti-6al-4v", "ti6al4v", "ti 6al 4v", "ti64", "grade 5"],
    "ss316l":    ["ss316l", "316l", "aisi 316l"],
    "ss17-4ph":  ["ss17-4ph", "17-4ph", "17-4 ph", "17-4 stainless", "17-4 precipitation hardening"],
    "in718":     ["in718", "inconel 718", "alloy 718"],
}

def _norm_material_name(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    key = s.strip().lower()
    if key in _MATERIAL_MODELS:
        return key
    for canon, aliases in _MATERIAL_SYNONYMS.items():
        if key == canon or key in aliases:
            return canon
    for canon, aliases in _MATERIAL_SYNONYMS.items():
        if any(a in key for a in aliases):
            return canon
    return None

def _is_nan(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def _validate_ranges_user_units(x: Dict[str, Optional[float]]) -> Dict[str, Any]:
    critical_missing = []
    for req in ("velocity", "power"):
        if _is_nan(x.get(req)):
            critical_missing.append(req)
    ood = []
    for k, (lo, hi) in _RANGES_USER_UNITS.items():
        v = x.get(k)
        if _is_nan(v):
            continue
        if v < 0:
            ood.append(k); continue
        if v < lo and (lo - v) / max(lo, 1e-12) > 0.5:
            ood.append(k)
        if v > hi and (v - hi) / max(hi, 1e-12) > 0.5:
            ood.append(k)
    return {
        "flags": {
            "critical_missing": len(critical_missing) > 0,
            "ood_parameters": len(ood) > 0,
        },
        "offending_fields": {
            "critical_missing": critical_missing,
            "ood_parameters": ood,
        }
    }

def _normalize_label(lbl) -> str:
    if lbl is None:
        return "none"
    try:
        i = int(lbl)
        return ID_TO_LABEL.get(i, "none")
    except Exception:
        pass
    s = str(lbl).strip().lower()
    if s in {"none", "no defect", "no-defect"}: return "none"
    if s in {"lof", "lack-of-fusion", "lack_of_fusion", "lack of fusion"}: return "LoF"
    if s in {"balling"}: return "balling"
    if s in {"keyhole", "keyholing"}: return "keyhole"
    return "none"

def _load_artifact(path: str):
    if joblib is None or not os.path.exists(path):
        return None, None
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z -= np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)

def _named_proba_to_vec(named: Dict[str, float]) -> List[float]:
    """Map a {label: prob} dict to [none, LoF, balling, keyhole]."""
    return [float(named.get(lbl, 0.0)) for lbl in LABEL_ORDER]

def _predict_with_proba_robust(model, arr) -> Dict[str, Any]:
    """
    Try predict_proba → decision_function→softmax → OVO vote → predict.
    Returns dict with 'label' (canonical string) and 'proba_named' {label: p}.
    """
    proba_named = {k: 0.0 for k in LABEL_ORDER}

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(arr)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == len(proba):
                for c_raw, p in zip(classes, proba):
                    proba_named[_normalize_label(c_raw)] += float(p)
                lbl = max(proba_named, key=proba_named.get)
                return {"label": lbl, "proba_named": proba_named}
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            df = model.decision_function(arr)
            if df.ndim == 2 and df.shape[0] == 1:
                probs = _softmax(np.array(df[0]))
                classes = getattr(model, "classes_", list(range(len(probs))))
                for c_raw, p in zip(classes, probs):
                    proba_named[_normalize_label(c_raw)] += float(p)
                lbl = max(proba_named, key=proba_named.get)
                return {"label": lbl, "proba_named": proba_named}
        except Exception:
            pass

    if model.__class__.__name__ == "OneVsOneClassifier" and hasattr(model, "estimators_") and hasattr(model, "classes_"):
        try:
            classes = list(model.classes_)
            votes = np.zeros(len(classes), dtype=int)
            k = 0
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    est = model.estimators_[k]
                    pred = est.predict(arr)
                    pred_lbl = str(pred[0])
                    if pred_lbl in classes:
                        idx = classes.index(pred_lbl)
                    else:
                        if hasattr(est, "classes_") and len(est.classes_) == 2:
                            if hasattr(est, "decision_function"):
                                df = est.decision_function(arr)
                                idx_lbl = est.classes_[1] if df[0] > 0 else est.classes_[0]
                            else:
                                idx_lbl = est.classes_[0]
                            idx = classes.index(idx_lbl) if idx_lbl in classes else i
                        else:
                            idx = i
                    votes[idx] += 1
                    k += 1
            lbl = _normalize_label(classes[int(np.argmax(votes))])
            proba_named = {k: 0.0 for k in LABEL_ORDER}
            proba_named[lbl] = 1.0
            return {"label": lbl, "proba_named": proba_named}
        except Exception:
            pass

    try:
        pred = model.predict(arr)
        lbl = _normalize_label(pred[0] if len(pred) else None)
        proba_named[lbl] = 1.0
        return {"label": lbl, "proba_named": proba_named}
    except Exception:
        return {"label": "none", "proba_named": proba_named}


class ClassifyDefect_Material(BaseTool):
    """
    Unified LPBF defect classifier for materials {Ti-6Al-4V, SS316L, SS17-4PH, IN718}.
    Inputs: velocity [mm/s], power [W], beamDiameter [µm], layerThickness [µm].
    Label IDs: 0=none, 1=LoF, 2=balling, 3=keyhole.
    """
    name: str = "classify_defect_material"
    description: str = (
        "Unified LPBF defect classifier for materials {Ti-6Al-4V, SS316L, SS17-4PH, IN718}. "
        "Inputs: velocity [mm/s], power [W], beamDiameter [µm], layerThickness [µm]. "
        "Label IDs: 0=none, 1=LoF, 2=balling, 3=keyhole."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "wc_material": {
                "type": "string",
                "description": "Material (one of 'Ti-6Al-4V', 'SS316L', 'SS17-4PH', 'IN718'). "
            },
            "input_process_parameters": {
                "type": "object",
                "description": (
                    "Dict or JSON string: {'velocity': <mm/s>, 'power': <W>, 'beamDiameter': <µm>, 'layerThickness': <µm>, 'hatchSpacing': <µm>}."
                )
            }
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
            if not mat_key:
                return ToolResult(error=f"Unsupported material '{wc_material}'. Supported: Ti-6Al-4V, SS316L, SS17-4PH, IN718.")

            payload = {
                "material": mat_key,
                "velocity": float(x_raw.get("velocity", 0.0) or 0.0),
                "power": float(x_raw.get("power", 0.0) or 0.0),
                "beamDiameter": float(x_raw.get("beamDiameter", 0.0) or 0.0),
                "layerThickness": float(x_raw.get("layerThickness", 0.0) or 0.0),
            }

            import httpx
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post("http://localhost:8000/predict/defect", json=payload, timeout=5.0)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.RequestError:
                    return ToolResult(error="ML Service unavailable (connection refused). Please ensure app/ml_service.py is running.")
                except httpx.HTTPStatusError as e:
                    return ToolResult(error=f"ML Service error: {e.response.text}")

            def _calculate_entropy(probs: Dict[str, float]) -> float:
                values = np.array(list(probs.values()), dtype=float)
                s = values.sum()
                if s <= 0: return 0.0
                values = values / s
                values = values[values > 0]
                return float(-np.sum(values * np.log2(values)))

            def _ml_reliability_from_probs(ml_probs: Dict[str, float], is_ood: bool=False) -> float:
                classes = ("none","lof","balling","keyhole")
                values = np.array([ml_probs.get(c, 0.0) for c in classes], dtype=float)
                s = values.sum()
                if s <= 0: return 0.1
                values = values / s
                
                entropy = _calculate_entropy(ml_probs)
                K = len(classes)
                h_norm = entropy / np.log2(K) if K > 1 else 0.0
                
                p_sorted = np.sort(values)[::-1]
                margin = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) >= 2 else 1.0
                
                reliability = (1.0 - h_norm) * (0.5 + 0.5 * margin)
                if is_ood: reliability *= 0.5
                return float(np.clip(reliability, 0.05, 0.9))

            proba_named = data.get("proba_named", {})
            
            entropy = _calculate_entropy(proba_named)
            reliability = _ml_reliability_from_probs(proba_named, is_ood=False)
            
            data["entropy"] = entropy
            data["reliability"] = reliability

            return ToolResult(output=json.dumps(data, ensure_ascii=False))

        except Exception as e:
            return ToolResult(error=f"classify_defect_material failed: {str(e)}")
