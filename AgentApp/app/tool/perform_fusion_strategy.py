import json
from typing import Any, Dict

from app.tool.base import BaseTool, ToolResult
from amcore.labels import LABEL_ORDER
from amcore.fusion import (
    calculate_entropy,
    ml_reliability_from_probs,
    deterministic_fusion,
)
from amcore.prompts import build_supervisor_prompt_for_params


class PerformFusionStrategy(BaseTool):
    name: str = "perform_fusion_strategy"
    description: str = (
        "Performs deterministic fusion of ML probabilities and Knowledge-Driven "
        "predictions from the tool knowledge_retrieve_literature(). "
        "Returns the fused belief, suggested label, and the fully constructed "
        "supervisor prompt."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "process_params": {
                "type": "object",
                "description": ("Dictionary containing 'material', 'Power', 'Velocity', "
                                "'beam_D' (or 'beam D'), 'layer_thickness', 'hatch_spacing'."),
            },
            "ml_probs": {
                "type": "object",
                "description": ("Dictionary of class probabilities e.g., "
                                "{'none': 0.1, 'keyhole': 0.9}. If provided as JSON string, it will be parsed."),
            },
            "rag_response": {
                "type": "string",
                "description": "Full text response from knowledge_retrieve_literature(), containing [BELIEF].",
            },
            "is_ood": {
                "type": "boolean",
                "description": "Whether this experimental condition is Out-of-Distribution (OOD). Default False.",
            },
        },
        "required": ["process_params", "ml_probs", "rag_response"],
    }

    async def execute(self,
                      process_params: Dict[str, Any],
                      ml_probs: Any,
                      rag_response: str,
                      is_ood: bool = False) -> ToolResult:
        try:
            if isinstance(ml_probs, str):
                try:
                    ml_probs = json.loads(ml_probs)
                except Exception:
                    return ToolResult(error="Invalid JSON string for ml_probs.")

            norm_params = dict(process_params)
            if "beam D" in norm_params and "beam_D" not in norm_params:
                norm_params["beam_D"] = norm_params["beam D"]

            exp_type = "out-of-distribution" if is_ood else "in-distribution"

            ml_entropy = calculate_entropy(ml_probs, classes=tuple(LABEL_ORDER))
            ml_reliability = ml_reliability_from_probs(
                ml_probs, is_ood, classes=tuple(LABEL_ORDER)
            )
            fused_belief, fused_label = deterministic_fusion(
                ml_probs_raw=ml_probs,
                ml_reliability=ml_reliability,
                rag_resp=rag_response,
                is_ood=is_ood,
            )
            prompt_sup = build_supervisor_prompt_for_params(
                process_params=norm_params,
                ml_probs=ml_probs,
                ml_entropy=ml_entropy,
                ml_reliability=ml_reliability,
                rag_response=rag_response,
                fused_belief=fused_belief,
                fused_label=fused_label,
                exp_type=exp_type,
            )
            output = {
                "fused_belief": fused_belief,
                "suggested_label": fused_label,
                "supervisor_prompt": prompt_sup,
            }
            return ToolResult(output=json.dumps(output, indent=2))
        except Exception as e:
            return ToolResult(error=f"Fusion Strategy Failed: {str(e)}")
