from app.tool.base import BaseTool, ToolResult
from typing import Any, Optional
from openai import OpenAI, AzureOpenAI
from app.config import config, LLMSettings


def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    """
    Return the merged LLMSettings for a profile (or 'default').
    Raises KeyError if the profile doesn't exist.
    """
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}")
    return profiles[profile]


def make_chat_client(profile: Optional[str] = "default") -> tuple[Any, str, dict]:

    llm = get_llm_settings(profile)
    api_type = llm.api_type

    if api_type == "azure":
        client = AzureOpenAI(
            api_key=llm.api_key,
            api_version=llm.api_version,
            azure_endpoint=llm.base_url,
        )
        deployment = llm.model
        default_kwargs = {
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, deployment, default_kwargs

    elif api_type == "Openai":
        client = OpenAI(api_key=llm.api_key)
        model = llm.model
        default_kwargs = {
            "model": model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }
        return client, model, default_kwargs

    else:
        raise ValueError(f"Unsupported api_type: {llm.api_type!r}")
# ----------------------------
# Helpers
# ----------------------------
system_prompt = """You are a keyword mining module for additive manufacturing literature search.

Your job:
Given one canonicalized payload describing an AM build condition,
you MUST output a JSON object with 4 keys:
  - "process_terms": list of strings
  - "material_terms": list of strings
  - "parameter_terms": list of strings
  - "objective_terms": list of strings

Semantics of each list:

1. process_terms:
   - Include the manufacturing process as given (e.g. "LPBF").
   - Expand to common synonymous / equivalent names.
     For LPBF / SLM / Powder Bed Fusion families, include:
       "LPBF", "Laser Powder Bed Fusion", "Selective Laser Melting",
       "Powder Bed Fusion", "SLM".
   - Never drop the original surface form(s), even if unknown.

2. material_terms:
   - Include the alloy / material name(s) provided AND all common aliases / trade names.
   - Examples:
     Ti-6Al-4V ⇒ ["Ti-6Al-4V", "Ti6Al4V", "Ti 6Al 4V", "Ti64", "Grade 5"]
     IN625 / Inconel 625 ⇒ ["IN625", "Inconel 625", "Alloy 625"]
     IN718 / Inconel 718 ⇒ ["IN718", "Inconel 718", "Alloy 718"]
     17-4PH ⇒ ["SS17-4PH", "17-4PH", "17-4 PH", "17-4 Stainless", "17-4 Precipitation Hardening"]
     316L ⇒ ["SS316L", "316L", "AISI 316L"]
   - If aliases/grade info appear in the payload (e.g. "aliases": [...]), include them.
   - Never leave this list empty if a material is present.

3. parameter_terms:
   - Look at the process parameter keys in the payload (e.g. laser_power, scan_velocity, beam_diameter, layer_thickness, hatch_spacing).
   - For each known canonical key, include its human search variants:
        laser_power      → "laser power", "power"
        scan_velocity    → "scan velocity", "scan speed"
        beam_diameter    → "beam diameter", "spot size", "laser spot size"
        layer_thickness  → "layer thickness", "powder layer thickness"
        hatch_spacing    → "hatch spacing", "hatch distance"
   - Also include the raw readable key names (e.g. "laser_power" → "laser power") to guarantee recall.
   - If there are unrecognized parameter names, include those names in human-readable form anyway.
   - Never return an empty list if any process parameters exist.

4. objective_terms:
   - Look at "prediction_objective".
   - If it's "classification": include typical defect-related targets:
        ["keyhole", "lack of fusion", "porosity"]
   - If it's "regression": include typical melt pool geometry targets:
        ["melt pool depth", "melt pool width", "melt pool length"]
   - If it's "both", unknown, or missing: include the UNION of both lists.

CRITICAL RULES:
- Output MUST be valid JSON only. Do not include markdown, code fences, or explanations.
- Each value MUST be a JSON array of strings (even if only one string).
- De-duplicate within each array.
- Preserve important capitalization (e.g. "LPBF", "Ti-6Al-4V", "Inconel 718").
- NEVER return an empty array when the payload clearly has relevant info for that category.
"""

def _build_user_prompt(payload_json: str) -> str:
    return f"""Extract keyword arrays from the following canonicalized payload.
            PAYLOAD:
            {payload_json}
            Return ONLY the JSON object with keys:
              "process_terms", "material_terms", "parameter_terms", "objective_terms".
            """

def llm_extract_keywords(
    payload_json: str,
    profile: Optional[str] = "default"
) -> str:
    """
    Call chat completion to extract keyword arrays.
    Returns the model's raw JSON string (txt) with:
      {
        "process_terms": [...],
        "material_terms": [...],
        "parameter_terms": [...],
        "objective_terms": [...]
      }

    We assume the system prompt forces strict JSON output
    and we DO NOT post-process or validate here.
    """

    client, model, default_kwargs = make_chat_client(profile=profile)

    system_prompt_local = system_prompt          # defined earlier
    user_prompt = _build_user_prompt(payload_json)  # defined earlier

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_local},
            {"role": "user", "content": user_prompt},
        ],
    )

    txt = resp.choices[0].message.content.strip()
    return txt

# ----------------------------
# MCP Tool
# ----------------------------
class TaskExtractKeywords(BaseTool):
    name: str = "task_extract_keywords"
    description: str = (
        "Extract structured search keywords (process/material/parameter terms) "
        "from a canonicalized AM payload, using an LLM. "
        "Returns a JSON object as a string with "
        'keys "process_terms", "material_terms", "parameter_terms", "objective_terms".'
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "payload_json": {
                "type": "string",
                "description": (
                    "JSON string produced by task_canonicalize_param that has "
                    "working_conditions, process_parameters, prediction_objective."
                ),
            }
        },
        "required": ["payload_json"]
    }

    async def execute(self, payload_json: str, **kwargs) -> ToolResult:
        # payload_json = "{\"working_conditions\": {\"process\": \"LPBF\", \"material\": {\"name\": \"Ti-6Al-4V\", \"aliases\": [\"Ti-6Al-4V\"], \"properties\": {\"Cp\": {\"value\": null, \"unit\": null}, \"k\": {\"value\": null, \"unit\": null}}}}, \"process_parameters\": {\"laser_power\": {\"value\": 300.0, \"unit\": \"W\"}, \"scan_velocity\": {\"value\": 1100.0, \"unit\": \"mm/s\"}, \"beam_diameter\": {\"value\": 100.0, \"unit\": \"\u0000b5m\"}, \"layer_thickness\": {\"value\": 30.0, \"unit\": \"\u0000b5m\"}, \"hatch_spacing\": {\"value\": 100.0, \"unit\": \"\u0000b5m\"}}, \"validity\": {\"status\": \"pass\", \"flags\": {\"critical_missing\": false, \"unit_unrecognized\": false, \"ood_process\": false, \"ood_material\": false, \"ood_parameters\": false}, \"offending_fields\": {\"critical_missing\": [], \"unit_unrecognized\": [], \"ood_process\": [], \"ood_material\": [], \"ood_parameters\": []}, \"ranges_used\": {\"laser_power\": {\"min\": 50.0, \"max\": 100.0, \"unit\": \"W\"}, \"scan_velocity\": {\"min\": 50.0, \"max\": 3000.0, \"unit\": \"mm/s\"}, \"beam_diameter\": {\"min\": 30.0, \"max\": 200.0, \"unit\": \"\u0000b5m\"}, \"layer_thickness\": {\"min\": 10.0, \"max\": 100.0, \"unit\": \"\u0000b5m\"}, \"hatch_spacing\": {\"min\": 40.0, \"max\": 200.0, \"unit\": \"\u0000b5m\"}}}, \"prediction_objective\": \"both\"}"
        try:
            txt = llm_extract_keywords(payload_json, profile="default")

            return ToolResult(output=txt)

        except Exception as e:
            return ToolResult(error=f"task_extract_keywords failed: {str(e)}")




