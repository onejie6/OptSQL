"""Run minimal non-secret health checks against both submission providers."""

from __future__ import annotations

import os

from openai import OpenAI


def _check(
    *,
    provider: str,
    env_name: str,
    base_url: str,
    model: str,
    extra_body: dict | None = None,
) -> None:
    api_key = (os.environ.get(env_name) or "").strip()
    if not api_key:
        raise RuntimeError(f"{env_name} is not set")
    try:
        response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Return exactly OPTSQL-OK and nothing else.",
                }
            ],
            max_tokens=32,
            temperature=0,
            extra_body=extra_body or {},
        )
    except Exception as exc:
        raise RuntimeError(
            f"{provider} preflight failed ({type(exc).__name__})"
        ) from None
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError(f"{provider} returned an empty response")
    usage = response.usage
    print(
        f"{provider}=ok model={model} "
        f"prompt_tokens={usage.prompt_tokens if usage else 'unknown'} "
        f"completion_tokens={usage.completion_tokens if usage else 'unknown'}"
    )


def main() -> None:
    _check(
        provider="openrouter",
        env_name="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-coder-plus",
    )
    _check(
        provider="deepseek",
        env_name="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        extra_body={"thinking": {"type": "disabled"}},
    )


if __name__ == "__main__":
    main()
