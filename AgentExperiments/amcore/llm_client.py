import os
from typing import Any, Optional, Tuple

from openai import OpenAI, AzureOpenAI

from app.config import config, LLMSettings


def get_llm_settings(profile: Optional[str] = None) -> LLMSettings:
    profiles = config.llm
    if profile is None:
        profile = "default"
    if profile not in profiles:
        raise KeyError(
            f"Unknown LLM profile '{profile}'. Available: {list(profiles.keys())}"
        )
    return profiles[profile]


def make_chat_client(profile: Optional[str] = "default") -> Tuple[Any, str, dict]:
    """Return (client, model_name, default_kwargs) for the given profile."""
    llm = get_llm_settings(profile)
    api_type = llm.api_type

    if api_type == "azure":
        client = AzureOpenAI(
            api_key=llm.api_key,
            api_version=llm.api_version,
            azure_endpoint=llm.base_url,
        )
        return client, llm.model, {
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }

    if api_type == "Openai":
        client = OpenAI(api_key=llm.api_key, base_url=llm.base_url)
        return client, llm.model, {
            "model": llm.model,
            "max_completion_tokens": llm.max_completion_tokens,
            "temperature": llm.temperature,
        }

    if api_type.lower() == "google":
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "Google GenAI package not installed. Please install it."
            ) from e

        api_key = llm.api_key or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        genai_config = types.GenerateContentConfig(
            temperature=llm.temperature,
            max_output_tokens=getattr(llm, "max_tokens", 8192),
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=1024,
            ),
        )
        return client, llm.model, {"config": genai_config}

    raise ValueError(f"Unsupported api_type: {llm.api_type!r}")


def call_llm(prompt: str,
             profile: Optional[str] = "default",
             system_msg: Optional[str] = None,
             reasoning_effort: str = "low",
             extra_kwargs: Optional[dict] = None) -> str:
    """Single-turn chat call. Handles Azure / OpenAI / Google GenAI uniformly.

    Args:
        reasoning_effort: GPT-5 / o1 style reasoning effort. Ignored by
            Google GenAI and other providers that don't expose this knob.
        extra_kwargs: per-call overrides merged into default_kwargs. Use
            this for sensitivity sweeps (e.g., {"temperature": 0.7} on
            non-GPT-5 profiles).
    """
    client, model, default_kwargs = make_chat_client(profile=profile)
    if system_msg is None:
        system_msg = (
            "You are an LPBF process analysis assistant and act as an LPBF "
            "defect classification model."
        )

    if "model" in default_kwargs:
        default_kwargs.pop("model")
    if extra_kwargs:
        default_kwargs.update(extra_kwargs)

    is_google = False
    try:
        from google import genai
        if isinstance(client, genai.Client):
            is_google = True
    except Exception:
        pass

    if is_google:
        full_prompt = f"{system_msg}\n\n{prompt}"
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            **default_kwargs,
        )
        final_text = []
        if (resp.candidates and resp.candidates[0].content
                and resp.candidates[0].content.parts):
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    final_text.append(part.text)
        out = "".join(final_text).strip()
        if not out:
            return "Error: Empty response from model (possibly thinking timeout or filter)."
        return out

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        reasoning_effort=reasoning_effort,
        **default_kwargs,
    )
    return resp.choices[0].message.content.strip()
