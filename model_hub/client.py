"""One client for every provider.

Raises on failure. See the package docstring for why: what a failed call means
for a score is a benchmark-design decision that belongs in the bundle, not here.

`generate_batch` exists because concurrency is the only thing that matters for
throughput against a vLLM server. Measured on 500 real ttct prompts, one L40S,
Qwen2.5-7B:

    concurrency  16  ->   628 tok/s  (-77.5% vs offline batch)
    concurrency  64  ->  1579 tok/s  (-43.4%)
    concurrency 256  ->  2751 tok/s  ( -1.5%)

vLLM's server defaults to --max-num-seqs 256, and the client has to keep that
many requests in flight for the scheduler to see the same queue depth an
offline `LLM.generate([...])` call gives it. DEFAULT_CONCURRENCY is therefore
tied to that default: lowering it does not fail, it just silently runs 4x
slower, which is the most easily-missed regression in this design.
"""
import os
from concurrent.futures import ThreadPoolExecutor

from .registry import resolve, ResolvedModel

DEFAULT_CONCURRENCY = int(os.environ.get("CREATIVITYPRISM_MODEL_CONCURRENCY", "256"))
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.75


class ProviderError(RuntimeError):
    """A provider call failed. Carries the original exception as __cause__."""


def _as_messages(prompt_or_messages):
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    return list(prompt_or_messages)


class Client:
    """Provider-agnostic generation.

    Accepts either a plain string or an OpenAI-style messages list, because the
    bundles disagree: aut passes messages, ttct and math_n_index pass a string.
    """

    def __init__(self, resolved: ResolvedModel, max_tokens=None, temperature=None,
                 top_p=None, top_k=None, seed=42, extra=None):
        self.model = resolved
        self.max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.extra = extra or {}
        self._impl = self._build()

    def _build(self):
        wire = self.model.wire
        if wire == "openai":
            from .providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider(self.model)
        if wire == "anthropic":
            from .providers.anthropic import AnthropicProvider
            return AnthropicProvider(self.model)
        if wire == "google":
            from .providers.google import GoogleProvider
            return GoogleProvider(self.model)
        raise ProviderError(f"No provider implementation for wire '{wire}'")

    def _params(self, overrides):
        p = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
        }
        p.update(self.extra)
        p.update({k: v for k, v in overrides.items() if v is not None})
        return p

    def generate(self, prompt_or_messages, **overrides):
        """One completion. Raises ProviderError on failure."""
        messages = _as_messages(prompt_or_messages)
        try:
            return self._impl.generate(messages, **self._params(overrides))
        except Exception as e:                       # noqa: BLE001 - re-raised, not swallowed
            raise ProviderError(
                f"{self.model.provider}/{self.model.model_id}: {e}"
            ) from e

    def complete(self, prompt, **overrides):
        """Raw completion -- no chat template applied.

        This is the exact equivalent of vLLM's offline ``LLM.generate(text)``,
        and it is what a task migrating off the offline API must use if its
        numbers are to stay comparable: chat wrapping changes the prompt the
        model actually sees. Only endpoints speaking the OpenAI wire format
        implement it; Anthropic and Google have no raw-completion route.
        """
        impl = getattr(self._impl, "complete", None)
        if impl is None:
            raise ProviderError(
                f"{self.model.provider} has no raw-completion endpoint; "
                f"use generate() (chat) instead."
            )
        try:
            return impl(prompt, **self._params(overrides))
        except Exception as e:                       # noqa: BLE001
            raise ProviderError(
                f"{self.model.provider}/{self.model.model_id}: {e}"
            ) from e

    def generate_batch(self, prompts, concurrency=None, return_exceptions=False,
                       raw=False, **overrides):
        """Concurrent completions, results in input order.

        `return_exceptions=True` yields the exception object in place of a
        result instead of aborting the batch -- which is what a caller that has
        to score every item wants.
        """
        n = concurrency or DEFAULT_CONCURRENCY
        n = max(1, min(n, len(prompts) or 1))

        call = self.complete if raw else self.generate

        def one(p):
            try:
                return call(p, **overrides)
            except Exception as e:                   # noqa: BLE001
                if return_exceptions:
                    return e
                raise

        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(one, prompts))


def get_client(canonical, bundle=None, keys_path=None, **kwargs):
    return Client(resolve(canonical, bundle=bundle, keys_path=keys_path), **kwargs)
