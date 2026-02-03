from app.tool.base import BaseTool, ToolResult
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Optional, Tuple, List, Any

# Constants matching benchmark
LABEL_ORDER = ["none", "lof", "balling", "keyhole"]

def _calculate_entropy(probs: Dict[str, float], classes=None) -> float:
    """Shannon entropy in bits."""
    if not probs:
        return 0.0
    if classes is None:
        values = np.array(list(probs.values()), dtype=float)
    else:
        values = np.array([probs.get(c, 0.0) for c in classes], dtype=float)

    s = values.sum()
    if s <= 0:
        return 0.0
    values = values / s  # ensure normalized

    values = values[values > 0]
    return float(-np.sum(values * np.log2(values)))

def _ml_reliability_from_probs(ml_probs: Dict[str, float], is_ood: bool,
                              classes=("none","lof","balling","keyhole")) -> float:
    values = np.array([ml_probs.get(c, 0.0) for c in classes], dtype=float)
    s = values.sum()
    if s <= 0:
        return 0.1
    values = values / s

    entropy = _calculate_entropy(ml_probs, classes=classes)
    K = len(classes)
    h_norm = entropy / np.log2(K) if K > 1 else 0.0  # 0..1

    p_sorted = np.sort(values)[::-1]
    margin = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) >= 2 else 1.0

    reliability = (1.0 - h_norm) * (0.5 + 0.5 * margin)

    if is_ood:
        reliability *= 0.5

    return float(np.clip(reliability, 0.05, 0.9))

def _extract_json_block(text: str, tag: str) -> Optional[Dict[str, float]]:
    """Extracts a JSON dict from within [TAG]...[/TAG]."""
    regex = re.compile(f"\[{tag}\]\s*(.+?)\s*\[/{tag}\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        content = m.group(1).replace("'", '"')
        return json.loads(content)
    except Exception:
        return None

def _extract_float_block(text: str, tag: str) -> Optional[float]:
    """Extracts a float from within [TAG]...[/TAG]."""
    regex = re.compile(f"\[{tag}\]\s*([0-9\.]+)\s*\[/{tag}\]", flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _deterministic_fusion(ml_probs_raw: Dict[str, float],
                          ml_reliability: float,
                          rag_resp: str,
                          is_ood: bool = False) -> Tuple[Dict[str, float], str]:
    """
    Performs Python-side deterministic fusion of agent outputs.
    Returns: (fused_belief_dict, fused_label)
    """
    default_belief = {l: 0.0 for l in LABEL_ORDER}
    
    ml_belief = ml_probs_raw.copy()
    ml_rel = ml_reliability
    
    rag_belief = _extract_json_block(rag_resp, "BELIEF") or default_belief
    rag_rel = _extract_float_block(rag_resp, "RELIABILITY")
    if rag_rel is None: rag_rel = 0.3 # default lower for RAG
    
    for b in [ml_belief, rag_belief]:
        total = sum(b.get(k, 0) for k in LABEL_ORDER)
        if total > 0:
            for k in LABEL_ORDER:
                b[k] = b.get(k, 0) / total
    
    if not is_ood:
        ml_rel *= 4.0

    fused_belief = {k: 0.0 for k in LABEL_ORDER}
    denom = ml_rel + rag_rel
    
    if denom > 0:
        for k in LABEL_ORDER:
            val = (ml_belief.get(k, 0.0) * ml_rel + rag_belief.get(k, 0.0) * rag_rel) / denom
            fused_belief[k] = val
    else:
        fused_belief = ml_belief.copy()

    fused_label = max(fused_belief, key=fused_belief.get)
    return fused_belief, fused_label


def _build_supervisor_prompt(process_params: Dict[str, Any],
                             ml_probs: Dict[str, float],
                             ml_entropy: float,
                             ml_reliability: float,
                             rag_response: str,
                             fused_belief: Dict[str, float],
                             fused_label: str,
                             exp_type: str = "in-distribution") -> str:
    """
    Supervisor Agent: Senior Engineer.
    Validates and explains the deterministically fused results.
    """
    material = process_params.get("material", "unknown material")
    power = process_params.get("Power", "unknown")
    velocity = process_params.get("Velocity", "unknown")
    beam_d = process_params.get("beam_D", "unknown") # Normalized spelling
    layer_t = process_params.get("layer_thickness", "unknown")
    hatch = process_params.get("hatch_spacing", "unknown")

    # Helper to check if unit is already there
    def fmt(val, unit):
        s = str(val)
        return s if unit in s else f"{s} {unit}"

    power_str = fmt(power, "W")
    velocity_str = fmt(velocity, "mm/s")
    beam_diam_str = fmt(beam_d, "µm")
    layer_thickness_str = fmt(layer_t, "µm")
    hatch_str = fmt(hatch, "µm")

    # Format belief for prompt
    f_belief_str = ", ".join([f'"{k}": {v:.2f}' for k,v in fused_belief.items()])

    # ML Output String
    none_p = ml_probs.get("none", 0.0)
    lof_p = ml_probs.get("lof", 0.0)
    ball_p = ml_probs.get("balling", 0.0)
    key_p = ml_probs.get("keyhole", 0.0)

    ml_output_str = (
        f"- \"none\": {none_p:.4f}\n- \"LoF\": {lof_p:.4f}\n- \"balling\": {ball_p:.4f}\n- \"keyhole\": {key_p:.4f}\n"
        f"- Prediction Entropy: {ml_entropy:.2f}\n"
        f"- Calculated Reliability: {ml_reliability:.2f}\n"
        f"Note: This model was trained on {exp_type} data."
    )

    prompt = (
        "You are a Senior AM Process Engineer (Supervisor). "
        "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
        f"that arise in {material} manufactured at {power_str}, utilizing a {beam_diam_str} beam, "
        f"traveling at {velocity_str}, with a layer thickness of {layer_thickness_str} and hatch spacing of {hatch_str}. "
        f"Specifically, consider whether these parameters respect the typical process window for {material}. "
        "Review the automated Probabilistic Fusion analysis and provide the final decision.\n\n"
        "Agent 1 (Data-Driven ML Analyst) - DIRECT PREDICTIONS:\n"
        f"{ml_output_str}\n\n"
        "Agent 2 (Knowledge-Driven Analyst):\n"
        f"{rag_response}\n\n"
        "--- FUSION ENGINE OUTPUT ---\n"
        f"Computed Fused Belief: {{{f_belief_str}}}\n"
        f"Suggested Label: {fused_label}\n"
        "----------------------------\n\n"
        "Task:\n"
        "1. Start by outputting the [FUSED_BELIEF] exactly as computed above.\n"
        "2. Review the Suggested Label. Does it align with the Agent evidence?\n"
        "   - Use physical indicators (VED, Overlap) only as a sanity check for extreme outliers.\n"
        "   - Do NOT override the fused result unless parameters are physically impossible for the predicted defect (e.g. keyhole at zero power).\n"
        "   - Otherwise, respect the fusion.\n"
        "3. Provide a Defect Risk Profile and Safe Adjustment recommendation.\n"
        "4. Conclude with the final [LABEL].\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {validation of fusion and mechanistic check} [/THINK]\n"
        "[FUSED_BELIEF] {\"none\": 0.X, \"lof\": 0.X, \"balling\": 0.X, \"keyhole\": 0.X} [/FUSED_BELIEF]\n"
        "[DEFECT_RISK_PROFILE] {summary of risk} [/DEFECT_RISK_PROFILE]\n"
        "[SAFE_ADJUSTMENT] {recommendation} [/SAFE_ADJUSTMENT]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt

class PerformFusionStrategy(BaseTool):
    name: str = "perform_fusion_strategy"
    description: str = (
        "Performs deterministic fusion of ML probabilities and Knowledge-Driven predictions from the tool knowledge_retrieve_literature()"
        "Returns the fused belief, suggested label, logic, and the fully constructed prompt "
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "process_params": {
                "type": "object",
                "description": "Dictionary containing 'material', 'Power', 'Velocity', 'beam_D' (or 'beam D'), 'layer_thickness', 'hatch_spacing'.",
            },
            "ml_probs": {
                "type": "object",
                "description": "Dictionary of class probabilities e.g., {'none': 0.1, 'keyhole': 0.9}. " 
                               "If provided as JSON string, it will be parsed.",
            },
            "rag_response": {
                "type": "string",
                "description": "Full text response from knowledge_retrieve_literature(), containing [BELIEF]",
            },
            "is_ood": {
                "type": "boolean",
                "description": "Whether this experimental condition is Out-of-Distribution (OOD). Default False."
            }
        },
        "required": ["process_params", "ml_probs", "rag_response"]
    }

    async def execute(self, process_params: Dict[str, Any], ml_probs: Any, rag_response: str, is_ood: bool = False) -> ToolResult:
        try:
            # Clean/Parse parameters
            if isinstance(ml_probs, str):
                try:
                    ml_probs = json.loads(ml_probs)
                except:
                    return ToolResult(error="Invalid JSON string for ml_probs.")

            # Normalize parameter keys if needed
            # e.g. "beam D" vs "beam_D"
            norm_params = {}
            for k, v in process_params.items():
                norm_params[k] = v
                if k == "beam D": norm_params["beam_D"] = v
                if k == "beam_D": norm_params["beam_D"] = v # ensure preference
            
            # Calculate ML Stats
            exp_type = "out-of-distribution" if is_ood else "in-distribution"
            
            ml_entropy = _calculate_entropy(ml_probs, classes=tuple(LABEL_ORDER))
            # Note: passing is_ood to reliability calculation as per benchmark logic
            ml_reliability = _ml_reliability_from_probs(ml_probs, is_ood, classes=tuple(LABEL_ORDER))

            # Perform Fusion
            fused_belief, fused_label = _deterministic_fusion(
                ml_probs_raw=ml_probs,
                ml_reliability=ml_reliability,
                rag_resp=rag_response,
                is_ood=is_ood
            )

            # Build Prompt
            prompt_sup = _build_supervisor_prompt(
                process_params=norm_params,
                ml_probs=ml_probs,
                ml_entropy=ml_entropy,
                ml_reliability=ml_reliability,
                rag_response=rag_response,
                fused_belief=fused_belief,
                fused_label=fused_label,
                exp_type=exp_type
            )
            
            output = {
                "fused_belief": fused_belief,
                "suggested_label": fused_label,
                "supervisor_prompt": prompt_sup
            }
            
            return ToolResult(output=json.dumps(output, indent=2))

        except Exception as e:
            return ToolResult(error=f"Fusion Strategy Failed: {str(e)}")
