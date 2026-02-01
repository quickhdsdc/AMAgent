from app.tool.base import BaseTool, ToolResult
import os, json, re
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ---------- Strict system prompt ----------
SYS_PROMPT = """
You perform task_partition_param(q), a STRICT parser for additive manufacturing (AM) queries.
Goal: extract working conditions, raw process parameters, and the prediction objective.

Rules:
1) Read a single user query about an AM job.
2) DO NOT invent values. If a field is absent, set it to null.
3) Determine the prediction objective:
   - If the query asks to classify defects (keywords: "defect", "keyhole", "lack of fusion", "balling"), set "classification".
   - If it asks for melt-pool geometry or other numeric quality metrics (keywords: "melt pool", "length", "width", "depth", "dimension"), set "regression".
   - If it says "predict quality", "expected quality", or is ambiguous, set "both".
   - If the user explicitly specifies one, use that.
4) Return ONLY a single valid JSON object following this schema. No extra text, no markdown, no code fences.

Target JSON schema (types and required keys):
{
  "working_conditions": {
    "process": string|null,
    "material": {
      "name": string|null,
      "properties": {
        "Cp": {"value": number|null, "unit": string|null},
        "k":  {"value": number|null, "unit": string|null}
      }
    }
  },
  "process_parameters_raw": {
    "laser_power":     {"value": number|null, "unit": string|null},
    "scan_velocity":   {"value": number|null, "unit": string|null},
    "beam_diameter":   {"value": number|null, "unit": string|null},
    "layer_thickness": {"value": number|null, "unit": string|null},
    "hatch_spacing":   {"value": number|null, "unit": string|null}
  },
  "prediction_objective": "regression" | "classification" | "both"
}

Important:
- Keep units exactly as given in the query (do not convert).
- Parse numeric values where possible; otherwise set value=null.
- Include ALL keys shown above even if values are null.
- Output ONLY the JSON object, nothing else.
""".strip()


def _strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.DOTALL)


def _extract_json(s: str) -> str:
    """
    Try to extract the first balanced JSON object from text.
    Falls back to stripped text if already looks like JSON.
    """
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # find the first { ... } block naively (balanced braces)
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return s


def _coerce_number_or_null(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            # tolerate spaces like " 250 "
            return float(x.strip())
        except Exception:
            return None
    return None


def _ensure_param(obj: Dict[str, Any], key: str):
    param = obj.get(key) or {}
    if not isinstance(param, dict):
        param = {}
    val = _coerce_number_or_null(param.get("value"))
    unit = param.get("unit") if isinstance(param.get("unit"), str) else (None if param.get("unit") is None else str(param.get("unit")))
    obj[key] = {"value": val, "unit": unit}


def _ensure_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist and types conform to the expected schema."""
    wc = d.get("working_conditions") or {}
    if not isinstance(wc, dict):
        wc = {}
    process = wc.get("process")
    if process is not None and not isinstance(process, str):
        process = str(process)
    material = wc.get("material") or {}
    if not isinstance(material, dict):
        material = {}
    m_name = material.get("name")
    if m_name is not None and not isinstance(m_name, str):
        m_name = str(m_name)
    m_props = material.get("properties") or {}
    if not isinstance(m_props, dict):
        m_props = {}

    # Material properties Cp, k
    for prop_key in ("Cp", "k"):
        prop = m_props.get(prop_key) or {}
        if not isinstance(prop, dict):
            prop = {}
        prop_val = _coerce_number_or_null(prop.get("value"))
        prop_unit = prop.get("unit") if isinstance(prop.get("unit"), str) else (None if prop.get("unit") is None else str(prop.get("unit")))
        m_props[prop_key] = {"value": prop_val, "unit": prop_unit}

    material = {"name": m_name if m_name is not None else None, "properties": m_props}
    wc = {"process": process if process is not None else None, "material": material}

    # Process parameters
    ppr = d.get("process_parameters_raw") or {}
    if not isinstance(ppr, dict):
        ppr = {}
    for k in ("laser_power", "scan_velocity", "beam_diameter", "layer_thickness", "hatch_spacing"):
        _ensure_param(ppr, k)

    # Objective
    obj = d.get("prediction_objective")
    if obj not in ("regression", "classification", "both"):
        # best-effort fallback to "both" if model didn't follow instruction
        obj = "both"

    return {
        "working_conditions": wc,
        "process_parameters_raw": ppr,
        "prediction_objective": obj,
    }


class TaskPartitionParam(BaseTool):
    name: str = "task_partition_param"
    description: str = (
        "Parse a natural-language AM query into a structured JSON object as the output containing working conditions, "
        "raw process parameters (values + original units), and the prediction objective."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User's natural-language query about an AM job.",
            }
        },
        "required": ["query"]
    }

    async def execute(self, query: str, **kwargs) -> ToolResult:
        # query = "Given a part to be printed in IN625 (Cp=429J/kg·K, k=9.8W/m·K) using LPBF with laser power 120 W, scan velocity 500 mm/s, beam diameter 62 µm, and layer thickness 60 µm, please predict the expected quality."

        try:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
            # client = AzureOpenAI(
            #     api_key="e78fb0af217940948b13612ce9732393",
            #     azure_endpoint="https://fhgenie-api-ipa-genai4aas.openai.azure.com/",
            #     api_version="2023-05-15",
            # )

            resp = client.chat.completions.create(
                model="gpt-5-2025-08-07",
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": query},
                ]
            )

            raw_text = (resp.choices[0].message.content or "").strip()
            return ToolResult(output=raw_text)

        except Exception as e:
            msg = str(e)
            hint = None
            if "AZURE_OPENAI" in msg or "authentication" in msg.lower() or "invalid url" in msg.lower():
                hint = ("Check Azure OpenAI env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                        "AZURE_OPENAI_API_VERSION.")
            return ToolResult(error=f"task_partition_param failed: {msg}" + (f" | {hint}" if hint else ""))



