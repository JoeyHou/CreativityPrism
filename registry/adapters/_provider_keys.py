#!/usr/bin/env python3
"""Emit shell `export` statements for provider API keys read from api_keys.json.

Called only by `export_provider_keys` in _common.sh. Values are shell-quoted and
consumed via `eval`, so they never appear in argv or on disk. Keys already set in
the environment are left alone, and obvious placeholders are ignored.

Usage: python3 _provider_keys.py <path-to-api_keys.json>
"""
import json
import os
import shlex
import sys

# Target env var -> the api_keys.json field names accepted for it, in priority order.
PROVIDER_FIELDS = {
    "OPENAI_API_KEY": ("OPENAI_API_KEY", "openai"),
    "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY", "anthropic"),
    "DEEPSEEK_API_KEY": ("DEEPSEEK_API_KEY", "deepseek"),
    # neocoder_dat reads GENAI_API_KEY; math_n_index accepts either spelling.
    "GENAI_API_KEY": ("GENAI_API_KEY", "GEMINI_API_KEY", "google", "gemini"),
    "GEMINI_API_KEY": ("GEMINI_API_KEY", "GENAI_API_KEY", "google", "gemini"),
}

PLACEHOLDER_PREFIXES = ("YOUR_", "<", "your_")


def usable(value):
    return (
        isinstance(value, str)
        and value.strip() != ""
        and value != "key"
        and not value.startswith(PLACEHOLDER_PREFIXES)
    )


def main():
    if len(sys.argv) < 2:
        return 0
    try:
        with open(sys.argv[1]) as fh:
            keys = json.load(fh)
    except (OSError, ValueError):
        return 0
    if not isinstance(keys, dict):
        return 0

    for env_var, fields in PROVIDER_FIELDS.items():
        if os.environ.get(env_var):
            continue
        for field in fields:
            value = keys.get(field)
            if usable(value):
                print(f"export {env_var}={shlex.quote(value)}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
