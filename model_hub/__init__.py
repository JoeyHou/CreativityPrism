"""model_hub -- one place that knows how to call a model.

Why this exists
---------------
The same provider-routing logic used to live in four near-identical copies
(`tasks/{aut_ttcw_cshort,ttct}/src/utils/api_wrapper.py`,
`tasks/math_n_index/api_warpper.py`, `tasks/neocoder_dat/src/models/model.py`).
Three of them shared a docstring verbatim, which is why the Gemini
thinking-budget fix landed in exactly one of them. All four chose a provider by
substring-matching the model name, so a model not named gpt/claude/gemini/
deepseek could not be evaluated at all and no custom endpoint could be reached.

Boundary: **model_hub raises, callers decide.**
------------------------------------------------
This package does transport and provider routing, and it *raises* on failure.
It deliberately does NOT decide what a failed call means for a score. Several
bundles record an API failure as a `NO` vote by returning the exception as a
string; that is a benchmark-design decision (see WORK_SUMMARY bug #5) and it
stays visible in the bundle that makes it, rather than being silently changed
here. Bundle shims therefore keep their own try/except and their own defaults.

Usage
-----
    from model_hub import resolve, get_client

    m = resolve("GPT4.1")                    # canonical name -> full config
    c = get_client("GPT4.1", max_tokens=512)
    c.generate("hello")                      # str or [{"role":..., "content":...}]
    c.generate_batch(prompts, concurrency=256)
"""
from .registry import (
    ResolvedModel,
    resolve,
    alias_for,
    list_models,
    ModelNotFound,
    AliasNotFound,
)
from .client import Client, get_client, ProviderError

__all__ = [
    "ResolvedModel", "resolve", "alias_for", "list_models",
    "ModelNotFound", "AliasNotFound",
    "Client", "get_client", "ProviderError",
]
