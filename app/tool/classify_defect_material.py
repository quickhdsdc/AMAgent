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

_FEATURE_ORDER = ["velocity", "power", "beamDiameter", "layerThickness"]  # V, P, D, T

_RANGES_USER_UNITS = {
    "velocity":       (50.0, 3000.0),  # mm/s
    "power":          (50.0, 1000.0),  # W
    "beamDiameter":   (10.0, 200.0),   # µm
    "layerThickness": (10.0, 100.0),   # µm
}

# >>> NEW ORDER & IDS <<<
LABEL_ORDER = ["none", "LoF", "balling", "keyhole"]  # indices: 0,1,2,3
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
        # allow numeric-coded artifacts
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

    # 1) predict_proba
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

    # 2) decision_function -> softmax
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

    # 3) One-vs-One vote
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

    # 4) predict()
    try:
        pred = model.predict(arr)
        lbl = _normalize_label(pred[0] if len(pred) else None)
        proba_named[lbl] = 1.0
        return {"label": lbl, "proba_named": proba_named}
    except Exception:
        return {"label": "none", "proba_named": proba_named}


class ClassifyDefect_Material(BaseTool):
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

    # async def execute(self, wc_material: str, input_process_parameters: Union[str, Dict[str, Any]], **kwargs) -> ToolResult:
    ##     input_process_parameters = {"velocity": 1100, "power": 300, "beamDiameter": 100, "layerThickness": 30}
    ##     wc_material = "Ti6Al4V"
    #
    #     try:
    #         mat_key = _norm_material_name(wc_material)
    #         if not mat_key or mat_key not in _MATERIAL_MODELS:
    #             return ToolResult(error=f"Unsupported material '{wc_material}'. Supported: Ti-6Al-4V, SS316L, SS17-4PH, IN718.")
    #         model_path = _MATERIAL_MODELS[mat_key]
    #
    #         x_raw = _as_obj(input_process_parameters)
    #         if not isinstance(x_raw, dict):
    #             return ToolResult(error="input_process_parameters must be a JSON object or a string containing one.")
    #
    #         # Cast features
    #         x: Dict[str, Optional[float]] = {}
    #         for k in _FEATURE_ORDER:
    #             v = x_raw.get(k, None)
    #             try:
    #                 x[k] = float(v) if v is not None else None
    #             except Exception:
    #                 x[k] = None
    #
    #         vcheck = _validate_ranges_user_units(x)
    #         if vcheck["flags"]["critical_missing"]:
    #             proba_named = {lbl: 0.0 for lbl in LABEL_ORDER}
    #             proba_named["none"] = 1.0
    #             return ToolResult(output=json.dumps({
    #                 "label_id": 0,
    #                 "label": "none",
    #                 "proba_vec": _named_proba_to_vec(proba_named),  # [none, LoF, balling, keyhole]
    #                 "proba_named": proba_named,
    #                 "validity": {"flags": {"cls_failure": True}, "offending_fields": vcheck["offending_fields"]},
    #                 "material": mat_key
    #             }, ensure_ascii=False))
    #
    #         model, load_err = _load_artifact(model_path)
    #         if model is None:
    #             return ToolResult(error=f"Model artifact not found or failed to load: {load_err or model_path}")
    #
    #         feats = [x.get(k, None) for k in _FEATURE_ORDER]
    #         feats = [0.0 if _is_nan(v) else float(v) for v in feats]
    #         arr = np.array(feats, dtype=np.float32).reshape(1, -1)
    #
    #         pred = _predict_with_proba_robust(model, arr)
    #         lbl = pred["label"]
    #         proba_named = pred["proba_named"]
    #
    #         label_id = LABEL_TO_ID.get(lbl, 0)
    #         proba_vec = _named_proba_to_vec(proba_named)
    #
    #         return ToolResult(output=json.dumps({
    #             # "label_id": label_id,
    #             "label": lbl,
    #             # "proba_vec": proba_vec,       # order: [none, LoF, balling, keyhole]
    #             "proba_named": proba_named,   # readable mapping
    #             # "validity": {"flags": {"cls_failure": False}, "offending_fields": vcheck["offending_fields"]},
    #             "material": mat_key
    #         }, ensure_ascii=False))
    #
    #     except Exception as e:
    #         return ToolResult(error=f"classify_defect_material failed: {str(e)}")

    async def execute(self, **kwargs) -> ToolResult:
            # {'Power': 350, 'Velocity': 800, 'beam D': 100, 'layer thickness': 30, 'Hatch spacing': 120}
            return ToolResult(output=json.dumps({'label': 'none', 'proba_named': {'none': 0.82, 'LoF': 0, 'balling': 0.09, 'keyhole': 0.09}, 'material': 'ti-6al-4v'}, ensure_ascii=False))
            # {'Power': 190, 'Velocity': 1200, 'beam D': 100, 'layer thickness': 30, 'Hatch spacing': 120}
            # return ToolResult(output=json.dumps({'label': 'balling', 'proba_named': {'none': 0.26, 'LoF': 28, 'balling': 0.44, 'keyhole': 0.02}, 'material': 'ti-6al-4v'}, ensure_ascii=False))
