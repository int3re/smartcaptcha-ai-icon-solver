"""Version-tolerant parsing of flet_charts events (attribute names differ between releases)."""

from __future__ import annotations

import json
from typing import Any

PAYLOAD_INDEX_KEYS = (
    "spot_index",
    "group_index",
    "bar_group_index",
    "point_index",
    "selected_index",
    "index",
)


def to_int(value: Any) -> int:
    if isinstance(value, bool):
        return -1
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return -1
    return -1


def event_payload_dict(e: Any, max_len: int = 2048) -> dict[str, Any] | None:
    data_raw = getattr(e, "data", None)
    if not isinstance(data_raw, str) or not data_raw:
        return None
    if len(data_raw) > max_len:
        return None
    if not (data_raw.startswith("{") and data_raw.endswith("}")):
        return None
    try:
        payload = json.loads(data_raw)
        return payload if isinstance(payload, dict) else None
    except ValueError:
        return None


def event_index(e: Any, *keys: str, allow_payload: bool = True) -> int:
    for key in keys:
        value = to_int(getattr(e, key, None))
        if value >= 0:
            return value

    if not allow_payload:
        return -1

    payload = event_payload_dict(e)
    if isinstance(payload, dict):
        candidate_keys = tuple(keys) + PAYLOAD_INDEX_KEYS
        for key in candidate_keys:
            value = to_int(payload.get(key))
            if value >= 0:
                return value
        for nested in payload.values():
            if isinstance(nested, dict):
                for key in candidate_keys:
                    value = to_int(nested.get(key))
                    if value >= 0:
                        return value
    return -1


def event_type(e: Any) -> str:
    for key in ("event_type", "type", "name"):
        value = getattr(e, key, None)
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    payload = event_payload_dict(e)
    if isinstance(payload, dict):
        for key in ("event_type", "type", "name"):
            value = payload.get(key)
            if value is not None:
                text = str(value).strip().lower()
                if text:
                    return text
    return ""


def is_chart_interaction_event(e: Any) -> bool:
    et = event_type(e)
    if not et:
        return True
    return any(token in et for token in ("hover", "tap", "click", "pointer"))


def is_click_like_event(e: Any) -> bool:
    raw = getattr(e, "event_type", None)
    if raw is None:
        raw = getattr(e, "type", None)
    if raw is None:
        raw = getattr(e, "name", None)
    et = str(raw).strip().lower() if raw is not None else ""
    if not et:
        return False
    if "hover" in et or "move" in et:
        return False
    return any(token in et for token in ("click", "tap", "up", "selected", "select"))


def event_short_dump(e: Any) -> str:
    et = event_type(e) or "unknown"
    idx = event_index(e, "spot_index", "group_index", "bar_group_index", "index", allow_payload=False)
    data_raw = getattr(e, "data", None)
    data_hint = f"len={len(data_raw)}" if isinstance(data_raw, str) else "-"
    return f"type={et}, idx={idx}, data={data_hint}"
