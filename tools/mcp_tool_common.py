"""Small pure helpers shared by the tools.mcp_tool_* modules: SDK 1.x/2.x field access,
error-text sanitising, numeric/bool coercion, timeouts and jitter. No origin state."""

import hashlib
import json
import logging
import math
import os
import random
import re
from typing import Any, Optional

logger = logging.getLogger("tools.mcp_tool")


class _OriginProxy:
    """Attribute proxy for ``tools.mcp_tool`` resolved at access time. The split modules read
    origin state (``_servers``, ``_lock``, SDK symbols, patchable helpers) through this so
    ``mock.patch("tools.mcp_tool.X")`` and origin-side rebinds stay effective, and so no split
    module needs the origin imported first (the origin imports them while initialising)."""

    __slots__ = ()

    def __getattr__(self, name: str):
        from tools import mcp_tool
        return getattr(mcp_tool, name)


_core = _OriginProxy()
_MISSING = object()


def mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across the 1.x -> 2.x rename to snake_case. Pydantic aliases
    don't apply to attribute access, so ``getattr(result, "isError", False)`` silently returns
    the default on 2.x — failed calls read as successful, schemas as empty."""
    value = getattr(obj, snake, _MISSING)
    if value is _MISSING:
        value = getattr(obj, camel, _MISSING)
    return default if value is _MISSING else value


def mcp_tool_registration_fingerprint(tools: list[Any]) -> str:
    """Fingerprint the MCP fields that affect Hermes tool registration.

    The MCP SDK absorbs ``outputSchema`` into its own result validator while
    ``tools/list`` runs. Hermes only needs to republish a tool when the name,
    description, input schema, or annotations used by registration change.
    """
    payload = []
    for tool in tools or []:
        if isinstance(tool, dict):
            value = tool
        else:
            value = None
            for method_name in ("model_dump", "dict"):
                method = getattr(tool, method_name, None)
                if callable(method):
                    try:
                        value = method(mode="json", by_alias=True)
                    except TypeError:
                        try:
                            value = method(by_alias=True)
                        except TypeError:
                            value = method()
                    except Exception:
                        value = None
                    if isinstance(value, dict):
                        break
            if not isinstance(value, dict):
                value = {
                    "name": mcp_field(tool, "name", "name"),
                    "description": mcp_field(tool, "description", "description"),
                    "inputSchema": mcp_field(tool, "input_schema", "inputSchema"),
                    "annotations": mcp_field(tool, "annotations", "annotations"),
                }
        annotations = value.get("annotations")
        if isinstance(annotations, dict):
            # Registration currently consumes only this trust hint. Other
            # MCP annotation metadata is intentionally excluded so it cannot
            # churn the registry or prompt cache.
            read_only_hint = annotations.get("readOnlyHint", annotations.get("read_only_hint")) is True
        else:
            read_only_hint = getattr(annotations, "readOnlyHint", None) is True
        input_schema = value.get("inputSchema", value.get("input_schema"))
        if not isinstance(input_schema, dict):
            input_schema = {}
        description = value.get("description") or ""
        value = {
            "name": value.get("name"),
            "description": description,
            "inputSchema": input_schema,
            "annotations": {"readOnlyHint": read_only_hint},
        }
        payload.append(value)
    payload.sort(key=lambda item: str(item.get("name", "")) if isinstance(item, dict) else str(item))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_DEFAULT_TOOL_TIMEOUT = 300      # seconds for tool calls


def _resolve_tool_timeout(config: dict) -> float:
    """Per-server tool-call timeout. Precedence: ``mcp_servers.<name>.timeout`` >
    ``timeouts.mcp.tool_call`` > the 300s default; values are platform-clamped by
    ``resolve_timeout``."""
    per_server = config.get("timeout")
    if per_server is not None:
        return per_server
    try:
        from agent.deadline import resolve_timeout
        resolved = resolve_timeout("mcp.tool_call", default=_DEFAULT_TOOL_TIMEOUT)
        if resolved is not None:
            return resolved
    except Exception:
        logger.debug("mcp.tool_call timeout resolution failed", exc_info=True)
    return _DEFAULT_TOOL_TIMEOUT


# Jitter on reconnect backoff so servers that lost the same backend don't retry in lockstep.
_BACKOFF_JITTER = 0.2            # +/-20%


def _jittered(seconds: float) -> float:
    """``seconds`` with +/-20% uniform jitter, floored at 0."""
    return max(0.0, seconds * random.uniform(1.0 - _BACKOFF_JITTER, 1.0 + _BACKOFF_JITTER))


# Credential patterns to strip from error messages: GitHub PAT, OpenAI-style key, Bearer token,
# and ``token= / key= / API_KEY= / password= / secret=`` assignments.
_CREDENTIAL_PATTERN = re.compile(
    r"(?:ghp_[A-Za-z0-9_]{1,255}|sk-[A-Za-z0-9_]{1,255}|Bearer\s+\S+"
    r"|(?:token|key|API_KEY|password|secret)=[^\s&,;\"']{1,255})", re.IGNORECASE)


def _env_ref_name(ref: str) -> str:
    """Bare env-var name from a ``${...}`` body; strips a Cursor-style ``env:`` prefix."""
    ref = ref.strip()
    if ref.startswith("env:"):
        ref = ref[len("env:"):].strip()
    return ref


def _sanitize_error(text: str) -> str:
    """Replace credential-like patterns with [REDACTED] before text reaches the LLM."""
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", text)


def _exc_str(exc: BaseException) -> str:
    """Non-empty string for *exc*: some exceptions (``anyio.ClosedResourceError``) carry no
    message, so fall back to ``repr`` to keep diagnostics."""
    text = str(exc).strip()
    return text or repr(exc)


def _prepend_path(env: dict, directory: str) -> dict:
    """Prepend *directory* to env PATH if it is not already present."""
    updated = dict(env or {})
    if directory:
        parts = [part for part in updated.get("PATH", "").split(os.pathsep) if part]
        if directory not in parts:
            parts = [directory, *parts]
        updated["PATH"] = os.pathsep.join(parts) if parts else directory
    return updated


def _safe_numeric(value, default, coerce=int, minimum=1):
    """Coerce a config value (YAML strings included) to a number, clamped to *minimum*;
    *default* on failure or non-finite floats."""
    try:
        result = coerce(value)
        if isinstance(result, float) and not math.isfinite(result):
            return default
        return max(result, minimum)
    except (TypeError, ValueError, OverflowError):
        return default


_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})


def _parse_boolish(value: Any, default: bool = True) -> bool:
    """Parse a bool-like config value with safe fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
    logger.warning("MCP config expected a boolean-ish value, got %r; using default=%s", value, default)
    return default


def _get_lifecycle_seconds(config: dict, key: str) -> Optional[float]:
    """Optional positive lifecycle timeout from top-level/nested ``lifecycle`` config (``0``
    disables; negatives and non-numbers are warned about and ignored)."""
    raw = config.get(key)
    if raw is None and isinstance(config.get("lifecycle"), dict):
        raw = config["lifecycle"].get(key)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning("MCP config %s must be a number of seconds; ignoring %r", key, raw)
        return None
    if seconds < 0:
        logger.warning("MCP config %s must be positive; ignoring %r", key, raw)
        return None
    return seconds or None
