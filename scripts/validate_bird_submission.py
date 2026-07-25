"""Offline preflight for the DashScope plus DeepSeek BIRD submission."""

from __future__ import annotations

import argparse
import os
import re
import tomllib
from pathlib import Path


EXPECTED_PROFILES = {
    "qwen_dashscope": {
        "model": "qwen3-coder-plus",
        "base_url": (
            "https://ws-t04z7f9rl8wvthos.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        "api_key": "env:DASHSCOPE_API_KEY",
    },
    "deepseek_controller": {
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key": "env:DEEPSEEK_API_KEY",
    },
}
SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-or-v1-|sk-)[A-Za-z0-9._-]{20,}"
)
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".md",
    ".sh",
    ".ps1",
    ".json",
    ".jsonl",
    ".txt",
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "data",
    "workspace",
    "results",
    "tmp",
    "__pycache__",
    "submission_dist",
}


def _secret_findings(root: Path) -> list[Path]:
    findings: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name for name in directory_names if name not in SKIP_PARTS
        ]
        current_path = Path(current_root)
        for file_name in file_names:
            path = current_path / file_name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if SECRET_PATTERN.search(text):
                findings.append(path)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-missing-keys", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    errors: list[str] = []
    profiles = config.get("llm_profiles") or {}

    for profile_name, expected in EXPECTED_PROFILES.items():
        actual = profiles.get(profile_name)
        if not actual:
            errors.append(f"missing LLM profile: {profile_name}")
            continue
        for key, expected_value in expected.items():
            if str(actual.get(key) or "").rstrip("/") != expected_value.rstrip("/"):
                errors.append(
                    f"{profile_name}.{key} must be {expected_value!r}, "
                    f"got {actual.get(key)!r}"
                )

    if (config.get("run") or {}).get("default_llm_profile") != "qwen_dashscope":
        errors.append("run.default_llm_profile must be qwen_dashscope")
    if (config.get("dataset") or {}).get("split") != "test":
        errors.append("submission config must use the BIRD test split")

    for section in (
        "few_shot_index",
        "value_retrieval",
        "schema_linking",
        "sql_generation",
        "sql_revision",
        "sql_selection",
    ):
        if (config.get(section) or {}).get("llm_profile") != "qwen_dashscope":
            errors.append(f"{section}.llm_profile must be qwen_dashscope")

    if not args.allow_missing_keys:
        for env_name in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"):
            if not os.getenv(env_name):
                errors.append(f"{env_name} is not set")

    errors.extend(
        f"key-like secret found in {path.relative_to(root)}"
        for path in _secret_findings(root)
    )

    if errors:
        print("SUBMISSION PREFLIGHT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("SUBMISSION PREFLIGHT PASSED")
    print(
        "base_provider=https://ws-t04z7f9rl8wvthos.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1"
    )
    print("base_model=qwen3-coder-plus")
    print("controller_provider=https://api.deepseek.com")
    print("controller_model=deepseek-v4-pro")
    print("dataset=bird/test")
    print("gold_runtime_access=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
