"""Permissive JSON parsing for LLM outputs that may include code fences."""
from __future__ import annotations

import json
from typing import Any


def parse_llm_json(raw: Any) -> Any:
    """Strip code fences if present, then json.loads. Returns the raw input
    if not a string, an empty dict for empty strings, or the original text
    on parse failure."""
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return {}
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return raw
