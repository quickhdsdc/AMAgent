
import os
import json
import glob
import math
import joblib
import numpy as np
import xgboost as xgb
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------------------------------
# CONSTANTS & CONFIG
# -------------------------------------------------------------------------

FEATURE_ORDER = ["velocity", "power", "beamDiameter", "layerThickness"]
LABEL_ORDER = ["none", "LoF", "balling", "keyhole"]

_MATERIAL_CLF_MODELS = {
    "ti-6al-4v": "./ml_models/Cls/bestF1_Ti6Al4V_RF_H_V_P_D_T.joblib",
    "ss316l":    "./ml_models/Cls/bestF1_SS316L_RF_H_V_P_D_T.joblib",
    "ss17-4ph":  "./ml_models/Cls/bestF1_SS174PH_RF_H_V_P_D_T.joblib",
    "in718":     "./ml_models/Cls/bestF1_IN718_GP_H_V_P_D_T.joblib",
}

_MATERIAL_REG_MODELS = {
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

loaded_models: Dict[str, Any] = {}

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def _find_reg_model_file(base_no_ext: str):
    if os.path.exists(base_no_ext + ".joblib"):
        return base_no_ext + ".joblib", "joblib"
    if os.path.exists(base_no_ext + ".json"):
        return base_no_ext + ".json", "xgb_json"
    for pat, k in (("*.joblib", "joblib"), ("*.json", "xgb_json")):
        hits = glob.glob(base_no_ext + pat[1:])
        if hits:
            return hits[0], k
    return None, None

def _load_model_file(path: str, kind: str):
    if kind == "joblib":
        return joblib.load(path)
    elif kind == "xgb_json":
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model
    return None

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z -= np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)

def _normalize_label(lbl) -> str:
    if lbl is None: return "none"
    s = str(lbl).strip().lower()
    if s in {"none", "no defect", "0"}: return "none"
    if s in {"lof", "lack of fusion", "1"}: return "LoF"
    if s in {"balling", "2"}: return "balling"
    if s in {"keyhole", "3"}: return "keyhole"
    return "none"

def _predict_cls(model, arr: np.ndarray):
    """
    Returns (label, proba_named)
    """
    proba_named = {k: 0.0 for k in LABEL_ORDER}
    
    def update_probs(classes, probs):
        for c, p in zip(classes, probs):
            proba_named[_normalize_label(c)] += float(p)

    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(arr)[0]
            update_probs(getattr(model, "classes_", LABEL_ORDER), p)
            return max(proba_named, key=proba_named.get), proba_named
        except Exception as e:
            print(f"DEBUG: predict_proba failed: {e}")
            import traceback
            traceback.print_exc()

    if hasattr(model, "decision_function"):
        try:
            df = model.decision_function(arr)
            if df.ndim == 2 and df.shape[0] == 1:
                p = _softmax(np.array(df[0]))
                update_probs(getattr(model, "classes_", range(len(p))), p)
                return max(proba_named, key=proba_named.get), proba_named
        except Exception as e:
             print(f"DEBUG: decision_function failed: {e}")

    try:
        pred = model.predict(arr)[0]
        lbl = _normalize_label(pred)
        proba_named[lbl] = 1.0
        return lbl, proba_named
    except Exception as e:
        print(f"DEBUG: predict failed: {e}")
        return "none", proba_named

def _predict_reg(model, arr: np.ndarray):
    try:
        y = model.predict(arr)
        if y is None: return None
        y = np.ravel(y)
        return float(y[0]) if y.size > 0 else None
    except:
        return None

# -------------------------------------------------------------------------
# LIFESPAN & APP
# -------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML models...")
    
    for mat, path in _MATERIAL_CLF_MODELS.items():
        if os.path.exists(path):
            try:
                loaded_models[f"cls_{mat}"] = joblib.load(path)
                print(f"Loaded CLS for {mat}")
            except Exception as e:
                print(f"Failed to load CLS {mat}: {e}")
        else:
            print(f"CLS model not found: {path}")

    for mat, dict_targets in _MATERIAL_REG_MODELS.items():
        for tgt, base in dict_targets.items():
            fpath, kind = _find_reg_model_file(base)
            if fpath:
                try:
                    loaded_models[f"reg_{mat}_{tgt}"] = _load_model_file(fpath, kind)
                    print(f"Loaded REG {mat}/{tgt}")
                except Exception as e:
                    print(f"Failed to load REG {mat}/{tgt}: {e}")
            else:
                 print(f"REG model not found base: {base}")
    
    yield
    loaded_models.clear()

app = FastAPI(lifespan=lifespan)

# -------------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    material: str
    velocity: float
    power: float
    beamDiameter: float
    layerThickness: float
    hatchSpacing: Optional[float] = 0.0

@app.post("/predict/defect")
async def predict_defect(req: PredictionRequest):
    mat = req.material.lower()
    model = loaded_models.get(f"cls_{mat}")
    if not model:
        raise HTTPException(status_code=404, detail=f"No CLS model for {mat}")

    arr = np.array([[req.hatchSpacing, req.velocity, req.power, req.beamDiameter, req.layerThickness]], dtype=np.float32)
    lbl, proba = _predict_cls(model, arr)
    
    return {
        "label": lbl,
        "proba_named": proba,
        "material": mat
    }

@app.post("/predict/meltpool")
async def predict_meltpool(req: PredictionRequest):
    mat = req.material.lower()
    arr = np.array([[req.hatchSpacing, req.velocity, req.power, req.beamDiameter, req.layerThickness]], dtype=np.float32)
    
    res = {}
    targets = {"depth": "dep", "width": "wid", "length": "len"}
    
    for hname, short in targets.items():
        model = loaded_models.get(f"reg_{mat}_{short}")
        val = None
        if model:
            val = _predict_reg(model, arr)
        res[hname] = {"value": val, "unit": "µm"}

    return {
        "pred": res,
        "material": mat
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
