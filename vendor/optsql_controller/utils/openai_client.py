"""OpenAI-compatible SDK request helpers.

The current configuration supports multiple OpenAI-compatible backends. When no
explicit API key / base URL is passed, the endpoint is selected from the model
name via config.resolve_model_endpoint(...).
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import os
import time
from typing import Any, Iterator

from config import DS_MODEL
from config import resolve_model_endpoint


ChatMessage = dict[str, str]

_LLM_REQUEST_LOG: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "llm_request_log",
    default=None,
)
_LLM_REQUEST_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "llm_request_context",
    default={},
)


@contextmanager
def capture_llm_requests(
    log: list[dict[str, Any]] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Capture all wrapped LLM requests executed in the current context."""
    active_log = log if log is not None else []
    token = _LLM_REQUEST_LOG.set(active_log)
    try:
        yield active_log
    finally:
        _LLM_REQUEST_LOG.reset(token)


@contextmanager
def llm_request_context(**metadata: Any) -> Iterator[None]:
    """Attach request metadata for all wrapped LLM requests in the context."""
    current = dict(_LLM_REQUEST_CONTEXT.get() or {})
    current.update({key: value for key, value in metadata.items() if value is not None})
    token = _LLM_REQUEST_CONTEXT.set(current)
    try:
        yield
    finally:
        _LLM_REQUEST_CONTEXT.reset(token)


def _message_payload(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    payload: dict[str, Any] = {}
    for key in ("role", "content", "refusal", "tool_calls", "function_call"):
        if hasattr(message, key):
            payload[key] = getattr(message, key)
    return payload


def _usage_payload(response: Any) -> dict[str, int]:
    usage_raw = getattr(response, "usage", None)
    if usage_raw is None:
        return {}
    return {
        "prompt_tokens": getattr(usage_raw, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_raw, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_raw, "total_tokens", 0) or 0,
    }


def _log_llm_request(
    *,
    started_at: float,
    elapsed_ms: float,
    model: str,
    base_url: str | None,
    request_params: dict[str, Any],
    messages: list[ChatMessage],
    response: Any | None,
    error: Exception | None,
) -> None:
    active_log = _LLM_REQUEST_LOG.get()
    if active_log is None:
        return

    metadata = dict(_LLM_REQUEST_CONTEXT.get() or {})
    entry: dict[str, Any] = {
        "started_at": started_at,
        "elapsed_ms": round(elapsed_ms, 3),
        "model": model,
        "base_url": base_url,
        "params": {key: value for key, value in request_params.items() if key != "messages"},
        "messages": list(messages),
        "message_count": len(messages),
        **metadata,
    }

    if error is not None:
        entry["error"] = str(error)
        active_log.append(entry)
        return

    if response is None:
        active_log.append(entry)
        return

    choice = response.choices[0] if getattr(response, "choices", None) else None
    if choice is not None:
        entry["finish_reason"] = getattr(choice, "finish_reason", None)
        entry["message"] = _message_payload(choice.message)
        entry["raw_response"] = choice.message.content or ""
    entry["usage"] = _usage_payload(response)
    active_log.append(entry)


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
):
    """Create an OpenAI SDK client using model-routed defaults.

    The OpenAI package is imported lazily so modules can be imported before the
    optional SDK dependency is installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The OpenAI SDK is required. Install it with `pip install openai`."
        ) from exc

    default_api_key, default_base_url = resolve_model_endpoint(model or DS_MODEL)
    resolved_api_key = api_key or default_api_key
    resolved_base_url = base_url or default_base_url

    if not resolved_api_key:
        raise ValueError(
            "Missing API key for the selected model. Configure the matching "
            "environment variable in config.py."
        )

    if not resolved_base_url:
        raise ValueError(
            "Missing base URL for the selected model. Configure the matching "
            "environment variable in config.py."
        )

    return OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)


def request_chat_completion(
    messages: list[ChatMessage],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client=None,
    **kwargs,
):
    """Send a chat completion request through an OpenAI-compatible client."""
    if not messages:
        raise ValueError("messages must be a non-empty list.")

    resolved_model = model or DS_MODEL
    resolved_client = client or get_openai_client(
        api_key=api_key,
        base_url=base_url,
        model=resolved_model,
    )
    request_params = {
        "model": resolved_model,
        "messages": messages,
    }

    if temperature is not None:
        request_params["temperature"] = temperature

    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens

    request_params.update(kwargs)
    enable_thinking = os.getenv("DS_ENABLE_THINKING")
    if enable_thinking is not None and "extra_body" not in request_params:
        request_params["extra_body"] = {
            "enable_thinking": enable_thinking.strip().lower() in {"1", "true", "yes"}
        }

    started_at = time.time()
    started = time.perf_counter()
    _, default_base_url = resolve_model_endpoint(resolved_model)
    response = None
    error: Exception | None = None
    try:
        response = resolved_client.chat.completions.create(**request_params)
        return response
    except Exception as exc:
        error = exc
        raise
    finally:
        _log_llm_request(
            started_at=started_at,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            model=resolved_model,
            base_url=base_url or default_base_url,
            request_params=request_params,
            messages=messages,
            response=response,
            error=error,
        )


def request_chat_text(
    messages: list[ChatMessage],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client=None,
    **kwargs,
) -> str:
    """Send a chat request and return the first response message content."""
    response = request_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        client=client,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def request_prompt_text(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client=None,
    **kwargs,
) -> str:
    """Send a plain prompt as chat messages and return the response text."""
    if not prompt:
        raise ValueError("prompt must be a non-empty string.")

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return request_chat_text(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        client=client,
        **kwargs,
    )
