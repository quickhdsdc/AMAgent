import json
import re
from typing import Dict, Optional

from amcore.labels import VALID_LABELS

_LABEL_REGEX = re.compile(
    r"\[LABEL\]\s*([^\[\]]+?)\s*\[/LABEL\]",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_label_from_response(text: str) -> str:
    if not text or not isinstance(text, str):
        return "unknown"

    m = _LABEL_REGEX.search(text)
    if not m:
        return "unknown"

    raw = m.group(1).strip().lower()
    norm = re.sub(r"[\{\}\"\'\n\r]", "", raw).strip()
    if ":" in norm:
        norm = norm.split(":")[-1].strip()
    if norm in VALID_LABELS:
        return norm
    return "unknown"


def extract_json_block(text: str, tag: str) -> Optional[Dict[str, float]]:
    """Extract a JSON dict from within [TAG]...[/TAG]."""
    regex = re.compile(rf"\[{tag}\]\s*(.+?)\s*\[/{tag}\]",
                       flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        content = m.group(1).replace("'", '"')
        return json.loads(content)
    except Exception:
        return None


def extract_float_block(text: str, tag: str) -> Optional[float]:
    """Extract a float from within [TAG]...[/TAG]."""
    regex = re.compile(rf"\[{tag}\]\s*([0-9\.]+)\s*\[/{tag}\]",
                       flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_think_block(text: str) -> str:
    regex = re.compile(r"\[THINK\]\s*(.+?)\s*\[/THINK\]",
                       flags=re.IGNORECASE | re.DOTALL)
    m = regex.search(text)
    return m.group(1).strip() if m else ""
