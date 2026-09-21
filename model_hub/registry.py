"""Resolve a canonical model name into everything needed to call it.

`registry/models.yaml` is the single source of truth. This module adds two
things the old code did not have:

1. **The declared `provider:` field is actually read.** Every previous wrapper
   ignored it and substring-matched the model name instead, so `gpt-oss-120b`
   (open weights) routed to the OpenAI API and anything not named after one of
   four vendors raised ValueError.

2. **Alias fallback ("convention first").** A bundle alias is now optional.
   Missing ones fall back to `api_model_id` for API models and `hf_id` for open
   models, so adding a bundle no longer means editing all 23 model entries.
   An explicit alias still wins, which is what keeps the bundles that use
   non-standard spellings (`gpt_4.1`, `qwen-72B-instruct`) working.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO_ROOT / "registry" / "models.yaml"

# api_keys.json field names accepted for each provider, in priority order.
# Mirrors registry/adapters/_provider_keys.py so both agree on what a key
# file may look like; keys already in the environment win over the file.
PROVIDER_KEY_FIELDS = {
    "openai":        ("OPENAI_API_KEY", "openai"),
    "anthropic":     ("ANTHROPIC_API_KEY", "anthropic"),
    "deepseek":      ("DEEPSEEK_API_KEY", "deepseek"),
    "google":        ("GENAI_API_KEY", "GEMINI_API_KEY", "google", "gemini"),
    "openai_compat": ("OPENAI_API_KEY", "openai"),
}

# Providers that speak the OpenAI wire format, with their default endpoint.
OPENAI_COMPAT_BASE_URLS = {
    "openai": None,                                  # SDK default
    "deepseek": "https://api.deepseek.com",
    "vllm": "http://127.0.0.1:8000/v1",
}

PLACEHOLDER_PREFIXES = ("YOUR_", "<", "your_")


class ModelNotFound(KeyError):
    pass


class AliasNotFound(KeyError):
    pass


@dataclass
class ResolvedModel:
    canonical: str
    type: str                      # "api" | "open"
    provider: str                  # openai | anthropic | google | deepseek | openai_compat | vllm
    model_id: str                  # what the provider is actually asked for
    api_key: str | None = None
    base_url: str | None = None
    hf_id: str | None = None
    serving: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)

    @property
    def is_open(self):
        return self.type == "open"

    @property
    def wire(self):
        """Which wire protocol to speak: 'openai' | 'anthropic' | 'google'."""
        if self.provider in ("anthropic", "google"):
            return self.provider
        return "openai"


def _usable(value):
    return (
        isinstance(value, str)
        and value.strip() != ""
        and value != "key"
        and not value.startswith(PLACEHOLDER_PREFIXES)
    )


def _load_yaml(path=None):
    import yaml
    path = Path(path) if path else MODELS_YAML
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get("models", {}) or {}


def list_models(path=None):
    return _load_yaml(path)


def _keys_file(keys_path=None):
    p = keys_path or os.environ.get("CREATIVITYPRISM_API_KEYS") or (REPO_ROOT / "api_keys.json")
    try:
        with open(p) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def find_api_key(provider, model_id=None, keys_path=None):
    """Environment first, then the provider block, then a model-keyed entry.

    The model-keyed lookup is last and exists only because the older bundles
    write api_keys.json that way ("gpt-4.1-2025-04-14": "sk-..."). A provider
    block is the shape everything should move to.
    """
    for env_var in PROVIDER_KEY_FIELDS.get(provider, ()):
        if env_var.isupper() and _usable(os.environ.get(env_var)):
            return os.environ[env_var]

    keys = _keys_file(keys_path)
    for fieldname in PROVIDER_KEY_FIELDS.get(provider, ()):
        if _usable(keys.get(fieldname)):
            return keys[fieldname]
    if model_id and _usable(keys.get(model_id)):
        return keys[model_id]
    if model_id and _usable(keys.get(str(model_id).lower())):
        return keys[str(model_id).lower()]
    return None


def alias_for(canonical, bundle, models=None, path=None):
    """Per-bundle alias, with convention-first fallback.

    Order: explicit alias -> api_model_id / aliases.api_call -> hf_id.
    Raises AliasNotFound only when nothing can be derived at all.
    """
    models = models if models is not None else _load_yaml(path)
    if canonical not in models:
        raise ModelNotFound(f"Model '{canonical}' not in registry/models.yaml")
    entry = models[canonical] or {}
    aliases = entry.get("aliases") or {}

    if bundle in aliases and _usable(str(aliases[bundle])):
        return aliases[bundle]

    if entry.get("type") == "open":
        if _usable(entry.get("hf_id")):
            return entry["hf_id"]
    else:
        for candidate in (entry.get("api_model_id"), aliases.get("api_call")):
            if _usable(str(candidate) if candidate else None):
                return candidate

    raise AliasNotFound(
        f"Model '{canonical}' has no alias for bundle '{bundle}' and none could "
        f"be derived (needs an explicit alias, api_model_id/api_call, or hf_id)."
    )


def resolve(canonical, bundle=None, keys_path=None, path=None):
    """Canonical name -> ResolvedModel."""
    models = _load_yaml(path)
    if canonical not in models:
        raise ModelNotFound(
            f"Model '{canonical}' not in registry/models.yaml. "
            f"Known: {', '.join(sorted(models))}"
        )
    entry = models[canonical] or {}
    aliases = entry.get("aliases") or {}
    mtype = entry.get("type", "api")

    provider = entry.get("provider") or ("vllm" if mtype == "open" else "openai")
    # huggingface is a weights host, not a serving provider; open weights are
    # served by vLLM in this repo.
    if provider == "huggingface":
        provider = "vllm"

    if bundle:
        model_id = alias_for(canonical, bundle, models=models)
    elif mtype == "open":
        model_id = entry.get("hf_id") or canonical
    else:
        model_id = entry.get("api_model_id") or aliases.get("api_call") or canonical

    base_url = entry.get("base_url")
    if base_url is None:
        base_url = OPENAI_COMPAT_BASE_URLS.get(provider)
    # An explicit override always wins: this is what points every task at one
    # already-running vLLM server.
    base_url = os.environ.get("CREATIVITYPRISM_MODEL_BASE_URL") or base_url

    key_env = entry.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env and _usable(os.environ.get(key_env)) else None
    if api_key is None:
        api_key = find_api_key(provider, model_id, keys_path)
    if api_key is None and provider == "vllm":
        api_key = "EMPTY"          # vLLM's OpenAI server ignores it but the SDK requires one

    return ResolvedModel(
        canonical=canonical, type=mtype, provider=provider, model_id=model_id,
        api_key=api_key, base_url=base_url, hf_id=entry.get("hf_id"),
        serving=entry.get("serving") or {}, aliases=aliases,
    )
