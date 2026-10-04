from typing import Any, Dict, List, Optional

import pandas as pd

from amcore.data_io import META_COL


def get_val_with_unit(row: pd.Series, col: str, unit: str,
                     fallback: str = "unknown") -> str:
    if col in row and pd.notnull(row[col]):
        return f"{row[col]} {unit}"
    return f"{fallback} {unit}"


def _row_params_strings(row: pd.Series):
    material = (row[META_COL]
                if META_COL in row and pd.notnull(row[META_COL])
                else "unknown material")
    return {
        "material": material,
        "power": get_val_with_unit(row, "Power", "W"),
        "velocity": get_val_with_unit(row, "Velocity", "mm/s"),
        "beam_diameter": get_val_with_unit(row, "beam D", "µm"),
        "layer_thickness": get_val_with_unit(row, "layer thickness", "µm"),
        "hatch_spacing": get_val_with_unit(row, "Hatch spacing", "µm"),
    }


def build_kd_agent_prompt(row: pd.Series,
                          rag_context: Optional[List[str]] = None,
                          evidence_pack: Optional[str] = None) -> str:
    """Knowledge-Driven Analyst prompt.
    Outputs: [THINK], [ASSUMPTIONS], [RELIABILITY], [BELIEF], [LABEL].

    If `evidence_pack` is provided, it is rendered verbatim as the literature
    evidence block (structured table + window summary produced by
    `amcore.evidence_pack.build_evidence_pack`). Otherwise the legacy
    numbered-chunk path is used for backward compatibility.
    """
    p = _row_params_strings(row)
    prompt = (
        "You are a Knowledge-Driven LPBF process analysis assistant. Assess potential imperfections for "
        f"Laser Powder Bed Fusion printing of {p['material']} at {p['power']}, using a {p['beam_diameter']} beam "
        f"traveling at {p['velocity']}, with a layer thickness of {p['layer_thickness']} and hatch spacing of {p['hatch_spacing']}. "
        f"Consider whether these parameters respect the typical process window for {p['material']}. "
        "Predict the potential defect label using the supplied literature evidence, when available, and your internal physics knowledge.\n\n"
        "Do not assume a defect is present unless evidence strongly favors a defect.\n"
        "Retrieved Literature Evidence:\n"
    )

    if evidence_pack:
        prompt += evidence_pack + "\n"
    elif rag_context:
        for i, s in enumerate(rag_context, 1):
            prompt += f"[{i}] {s}\n"
    else:
        prompt += "No specific literature found.\n"

    if evidence_pack:
        evidence_guidance = (
            "   - The evidence above is a table of extracted literature datapoints. "
            "Compare all reported parameters and outcomes; a similar VED alone is not an exact match.\n"
        )
    elif rag_context:
        evidence_guidance = (
            "   - The evidence above consists of literature excerpts, not a row-level "
            "experimental table. Use only measurements and outcomes explicitly stated in them.\n"
        )
    else:
        evidence_guidance = "   - No literature was retrieved; do not claim a measured match or cited process window.\n"

    prompt += (
        "\nTask:\n"
        "1. Compare the target parameters with the evidence and your internal knowledge.\n"
        f"{evidence_guidance}"
        "2. Check for a material-specific process window only if one is reported. "
        "Treat an in-window setting as support for 'none', not as proof of a defect-free outcome.\n"
        "3. Assess your reliability:\n"
        "   - Use the retrieved evidence to validate your predictions. Direct matches increase reliability (0.7-1.0).\n"
        "   - If RAG evidence is missing or weak, rely on your internal physics knowledge. If your theoretical analysis is confident, you may assign MEDIUM to HIGH reliability (0.3-0.7).\n"
        "   - Only assign LOW reliability (0.1-0.3) if you lack both external evidence and internal theoretical confidence.\n"
        "4. List assumptions:\n"
        "   - Many papers focus on failures; do not assume a defect exists if parameters look nominal/standard.\n"
        "5. Estimate your belief distribution over defects, including 'none'.\n"
        "   - Unless evidence includes a clear mechanism indicating failure (e.g., very low overlap / extreme low energy, or explicit keyhole indicators), assign at least 0.1 probability to 'none'.\n"
        "6. Conclude with a single label.\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {literature comparison and knowledge inference} [/THINK]\n"
        "[ASSUMPTIONS] {list of assumptions and numeric mismatch warnings} [/ASSUMPTIONS]\n"
        "[RELIABILITY] {0.0 to 1.0} [/RELIABILITY]\n"
        "[BELIEF] {\"none\": 0.X, \"lof\": 0.X, \"balling\": 0.X, \"keyhole\": 0.X} [/BELIEF]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )
    return prompt


def build_supervisor_prompt(row: pd.Series,
                            ml_probs: Dict[str, float],
                            ml_entropy: float,
                            ml_reliability: float,
                            rag_response: str,
                            fused_belief: Dict[str, float],
                            fused_label: str,
                            exp_type: str = "in-distribution") -> str:
    """Supervisor / Senior Engineer prompt. Validates the deterministic fusion."""
    p = _row_params_strings(row)

    f_belief_str = ", ".join([f'"{k}": {v:.2f}' for k, v in fused_belief.items()])

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
        f"that arise in {p['material']} manufactured at {p['power']}, utilizing a {p['beam_diameter']} beam, "
        f"traveling at {p['velocity']}, with a layer thickness of {p['layer_thickness']} and hatch spacing of {p['hatch_spacing']}. "
        f"Specifically, consider whether these parameters respect the typical process window for {p['material']}. "
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


def build_kd_agent_prompt_binary(row: pd.Series,
                                rag_context: Optional[List[str]] = None,
                                evidence_pack: Optional[str] = None,
                                ) -> str:
    """Knowledge-Driven Analyst prompt for the binary good/defective task.
    Outputs: [THINK], [ASSUMPTIONS], [RELIABILITY], [BELIEF], [LABEL].
    [BELIEF] schema is {"good": X, "defective": X}.

    If `evidence_pack` is provided, it is rendered verbatim as the literature
    evidence block — a structured table of nearest-parameter datapoints plus
    a one-line window summary, produced by
    `amcore.evidence_pack.build_evidence_pack`. This is the path used by the
    parameter-grounded retrieval (rag_v2). When only `rag_context` is given
    (legacy chunk-list path), each entry is rendered as a numbered snippet.
    """
    p = _row_params_strings(row)
    prompt = (
        "You are a Knowledge-Driven LPBF process analysis assistant. Assess "
        f"whether the resulting part is likely to be defect-free (good) or to "
        f"contain significant defects (defective) for {p['material']} "
        f"manufactured at {p['power']}, utilizing a {p['beam_diameter']} beam, "
        f"traveling at {p['velocity']}, with a layer thickness of "
        f"{p['layer_thickness']} and hatch spacing of {p['hatch_spacing']}. "
        f"Specifically, consider whether these parameters respect the typical "
        f"process window for {p['material']}. A part is considered "
        f"`defective` if its relative density is below ~99 % or if any "
        f"significant defect mechanism (lack of fusion, balling, keyhole, "
        f"hot cracking, severe porosity) is expected to dominate.\n\n"
        "Do not assume a defect is present unless evidence strongly favors "
        "one. Retrieved Literature Evidence:\n"
    )

    if evidence_pack:
        prompt += evidence_pack + "\n"
    elif rag_context:
        for i, s in enumerate(rag_context, 1):
            prompt += f"[{i}] {s}\n"
    else:
        prompt += "No specific literature found.\n"

    prompt += (
        "\nTask (asymmetric evidence-weighted reasoning):\n"
        "Step A — Extract from retrieval:\n"
        "   If structured rows are supplied, read the table as extracted "
        "literature datapoints. If only numbered excerpts are supplied, "
        "use only measurements explicitly reported in those excerpts. "
        "If no evidence is supplied, do not claim a measured match.\n"
        "   (i) Any explicit optimal VED / (P, v, h, t) window cited for THIS "
        "material. Do not infer a material-specific window from examples or "
        "transfer one from another material.\n"
        "   (ii) Any DIRECT density measurement (with a reported numeric "
        "percentage) at parameters within ±15% of the query (power, velocity, "
        "hatch, layer thickness). Such a row dominates window-only "
        "reasoning.\n\n"
        "Step B — Primary decision:\n"
        "   - Compute query VED = P / (v · (h/1000) · (t/1000)) in J/mm^3 "
        "when h and t are in µm.\n"
        "   - If (ii) exists, BASE the prediction on the closest direct "
        "numerical match: reported density ≥ 99% → `good`; < 99% → "
        "`defective`. A direct match dominates window arguments.\n"
        "   - Otherwise apply ASYMMETRIC window reasoning:\n"
        "       * OUTSIDE the cited material-specific window (or a clear "
        "failure mechanism applies) → `defective` is the strong default "
        "(necessary-condition violation).\n"
        "       * INSIDE the cited window WITHOUT a direct match → only "
        "MODERATE support for `good`. Being in-window is necessary but not "
        "sufficient: density also depends on hatch overlap, scan strategy, "
        "atmosphere, and powder condition, none of which VED captures.\n\n"
        "Step C — Modulators (override Step B only on strong contradiction):\n"
        "   - Hatch / melt-pool overlap: if hatch spacing is large relative "
        "to the melt-pool width implied by line energy (P/v) and beam "
        "diameter, raise lack-of-fusion risk even with in-window VED.\n"
        "   - Treat a transition zone as uncertain only when supported by "
        "material-specific evidence.\n"
        "   - Scan strategy / atmosphere: cite only if retrieval flags them.\n\n"
        "Step D — Belief assignment matrix (pick the row that fits):\n"
        "   - Direct ≥99% match at near-identical params           "
        "→ good ≥ 0.75\n"
        "   - Direct <99%  match at near-identical params           "
        "→ defective ≥ 0.75\n"
        "   - In-window, no direct match, no modulator concern      "
        "→ good ≈ 0.55–0.65\n"
        "   - In-window but modulator flags risk (hatch overlap…)   "
        "→ defective ≈ 0.55–0.70\n"
        "   - Out-of-window, no contradicting direct match          "
        "→ defective ≥ 0.70\n"
        "   - Retrieval cites explicit failure mechanism here       "
        "→ defective ≥ 0.80\n"
        "   Always keep good ≥ 0.10 and defective ≥ 0.10.\n\n"
        "Step E — Reliability:\n"
        "   - HIGH   (0.75–0.95): a direct numerical match drives the call.\n"
        "   - MEDIUM (0.45–0.75): asymmetric window reasoning grounded in a "
        "cited material-specific window.\n"
        "   - LOW    (0.20–0.45): no material-specific window cited; relying "
        "on generic physics or cross-material analogy.\n\n"
        "Step F — Conclude with a single [LABEL].\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {Step A findings, Step B decision, Step C modulator check} "
        "[/THINK]\n"
        "[ASSUMPTIONS] {assumptions and numeric mismatch warnings} "
        "[/ASSUMPTIONS]\n"
        "[RELIABILITY] {0.0 to 1.0 per Step E} [/RELIABILITY]\n"
        "[BELIEF] {\"good\": 0.X, \"defective\": 0.X} [/BELIEF]\n"
        "[LABEL] {one of \"good\", \"defective\"} [/LABEL]"
    )
    return prompt


def build_supervisor_prompt_binary(row: pd.Series,
                                  ml_probs: Dict[str, float],
                                  ml_entropy: float,
                                  ml_reliability: float,
                                  rag_response: str,
                                  fused_belief: Dict[str, float],
                                  fused_label: str,
                                  exp_type: str = "in-distribution") -> str:
    """Supervisor prompt for the binary good/defective task."""
    p = _row_params_strings(row)
    f_belief_str = ", ".join([f'"{k}": {v:.2f}' for k, v in fused_belief.items()])

    good_p = ml_probs.get("good", 0.0)
    bad_p = ml_probs.get("defective", 0.0)

    ml_output_str = (
        f"- \"good\":      {good_p:.4f}\n"
        f"- \"defective\": {bad_p:.4f}\n"
        f"- Prediction Entropy: {ml_entropy:.2f}\n"
        f"- Calculated Reliability: {ml_reliability:.2f}\n"
        f"Note: This model was trained on {exp_type} data."
    )

    return (
        "You are a Senior AM Process Engineer (Supervisor). Assess in detail "
        "whether the LPBF print is likely good (defect-free) or defective for "
        f"{p['material']} manufactured at {p['power']}, utilizing a "
        f"{p['beam_diameter']} beam, traveling at {p['velocity']}, with a "
        f"layer thickness of {p['layer_thickness']} and hatch spacing of "
        f"{p['hatch_spacing']}. A part is `defective` if expected relative "
        "density < ~99 % or any significant defect mechanism dominates. "
        "Review the automated Probabilistic Fusion analysis and provide the "
        "final decision.\n\n"
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
        "2. Review the Suggested Label. Does it align with the Agent "
        "evidence?\n"
        "   - Use physical indicators (VED, LED) only as a sanity check for "
        "extreme outliers.\n"
        "   - Do NOT override the fused result unless parameters are "
        "physically impossible for the predicted class.\n"
        "   - Otherwise, respect the fusion.\n"
        "3. Provide a Defect Risk Profile and Safe Adjustment recommendation.\n"
        "4. Conclude with the final [LABEL].\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {validation of fusion and mechanistic check} [/THINK]\n"
        "[FUSED_BELIEF] {\"good\": 0.X, \"defective\": 0.X} [/FUSED_BELIEF]\n"
        "[DEFECT_RISK_PROFILE] {summary of risk} [/DEFECT_RISK_PROFILE]\n"
        "[SAFE_ADJUSTMENT] {recommendation} [/SAFE_ADJUSTMENT]\n"
        "[LABEL] {one of \"good\", \"defective\"} [/LABEL]"
    )


def build_zs_prompt(row: pd.Series) -> str:
    """Pure zero-shot prompt used by the LLM-ZS baseline (no KS, no fusion)."""
    p = _row_params_strings(row)
    return (
        "You are a Knowledge-Driven LPBF process analysis assistant. Assess potential imperfections for "
        f"Laser Powder Bed Fusion printing of {p['material']} at {p['power']}, using a {p['beam_diameter']} beam "
        f"traveling at {p['velocity']}, with a layer thickness of {p['layer_thickness']} and hatch spacing of {p['hatch_spacing']}. "
        f"Consider whether these parameters respect the typical process window for {p['material']}. "
        "Predict the potential defect label using only these process parameters and your internal physics knowledge; no literature is supplied.\n\n"
        "Return ONLY the schema below:\n"
        "[THINK] {physics-based inference and uncertainty} [/THINK]\n"
        "[LABEL] {one of \"none\", \"lof\", \"balling\", \"keyhole\"} [/LABEL]"
    )


def build_supervisor_prompt_for_params(process_params: Dict[str, Any],
                                       ml_probs: Dict[str, float],
                                       ml_entropy: float,
                                       ml_reliability: float,
                                       rag_response: str,
                                       fused_belief: Dict[str, float],
                                       fused_label: str,
                                       exp_type: str = "in-distribution") -> str:
    """Supervisor prompt variant accepting a plain dict (for the live MCP tool).

    Mirrors build_supervisor_prompt() but does not depend on a pandas row.
    """
    material = process_params.get("material", "unknown material")
    power = process_params.get("Power", "unknown")
    velocity = process_params.get("Velocity", "unknown")
    beam_d = process_params.get("beam_D", process_params.get("beam D", "unknown"))
    layer_t = process_params.get("layer_thickness", "unknown")
    hatch = process_params.get("hatch_spacing", "unknown")

    def fmt(val, unit):
        s = str(val)
        return s if unit in s else f"{s} {unit}"

    p = {
        "material": material,
        "power": fmt(power, "W"),
        "velocity": fmt(velocity, "mm/s"),
        "beam_diameter": fmt(beam_d, "µm"),
        "layer_thickness": fmt(layer_t, "µm"),
        "hatch_spacing": fmt(hatch, "µm"),
    }

    f_belief_str = ", ".join([f'"{k}": {v:.2f}' for k, v in fused_belief.items()])
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

    return (
        "You are a Senior AM Process Engineer (Supervisor). "
        "Your task is to assess in detail the potential imperfections for Laser Powder Bed Fusion printing "
        f"that arise in {p['material']} manufactured at {p['power']}, utilizing a {p['beam_diameter']} beam, "
        f"traveling at {p['velocity']}, with a layer thickness of {p['layer_thickness']} and hatch spacing of {p['hatch_spacing']}. "
        f"Specifically, consider whether these parameters respect the typical process window for {p['material']}. "
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
