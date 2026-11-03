"""Parse Groq quota responses without storing request content or credentials."""

import json
import re


def duration_seconds(value):
    """Parse provider reset durations such as '2m59.56s' or Retry-After seconds."""
    text = str(value or "").strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    terms = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", text)
    if not terms or "".join(number + unit for number, unit in terms) != text:
        return None
    scale = {"ms": .001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    return sum(float(number) * scale[unit] for number, unit in terms)


def quota_failure(error):
    """Identify daily exhaustion separately from a short minute-rate wait."""
    response = getattr(error, "response", None)
    headers = {str(k).lower(): v for k, v in (getattr(response, "headers", {}) or {}).items()}
    body = getattr(error, "body", None)
    text = str(error) + " " + json.dumps(body, default=str)
    daily = bool(re.search(r"\b(?:RPD|TPD)\b|per[ _-]?day|daily", text, re.I))
    # Groq documents the request counter as requests/day, not requests/minute.
    daily = daily or str(headers.get("x-ratelimit-remaining-requests", "")).strip() == "0"
    retry = duration_seconds(headers.get("retry-after"))
    if daily:
        # Token reset headers describe TPM, so do not use them for daily tokens.
        daily_tokens = bool(re.search(r"\bTPD\b|tokens?\s+per\s+day", text, re.I))
        reset = None if daily_tokens else duration_seconds(headers.get("x-ratelimit-reset-requests"))
        if reset is None:
            match = re.search(r"try again in\s+([\d.hms]+)", text, re.I)
            reset = duration_seconds(match[1]) if match else None
        return {"daily": True, "retry_after": max(1.0, reset or retry or 86400)}
    reset = duration_seconds(headers.get("x-ratelimit-reset-tokens"))
    return {"daily": False, "retry_after": max(1.0, retry or reset or 60)}
