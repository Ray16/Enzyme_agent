"""Model-specific Google capabilities and quota reset times; no discovery hints."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import urllib.request

from enzyme_discovery_agent import config


def next_quota_reset(epoch=None):
    now = datetime.fromtimestamp(epoch, ZoneInfo("America/Los_Angeles")) if epoch is not None else datetime.now(ZoneInfo("America/Los_Angeles"))
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def fetch_model_metadata():
    """Read input/output limits using the campaign's key, without generating text."""
    request = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models",
                                     headers={"x-goog-api-key": config.GEMINI_API_KEY})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    return {row["name"].removeprefix("models/"): {
                "input_token_limit": row.get("inputTokenLimit"),
                "output_token_limit": row.get("outputTokenLimit"),
                "methods": row.get("supportedGenerationMethods", []), "context_source": "Google models API"}
            for row in data.get("models", [])}


def quota_values(error):
    """Extract numeric, model-specific limits; never guess a daily remaining balance."""
    body = getattr(error, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return {}
    result = {}
    def visit(value):
        if isinstance(value, dict):
            if str(value.get("@type", "")).endswith("google.rpc.QuotaFailure"):
                for row in value.get("violations", []):
                    try:
                        limit = int(row.get("quotaValue"))
                    except (ValueError, TypeError, AttributeError):
                        continue
                    identifier = "".join(c for c in str(row.get("quotaId", "")).lower() if c.isalpha())
                    if "inputtokens" in identifier and "perminute" in identifier:
                        result["input_tokens_per_minute"] = limit
                    elif "requests" in identifier and "perminute" in identifier:
                        result["requests_per_minute"] = limit
                    elif "requests" in identifier and "perday" in identifier:
                        result["requests_per_day"] = limit
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(body)
    return result
