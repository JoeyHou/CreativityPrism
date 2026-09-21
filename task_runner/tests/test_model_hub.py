"""model_hub unit tests. No network, no credentials, no GPU."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_hub import registry
from model_hub.registry import (
    resolve, alias_for, ModelNotFound, AliasNotFound, ResolvedModel,
)


class TestProviderIsDeclaredNotGuessed(unittest.TestCase):
    """The old wrappers substring-matched the model name. These lock in that
    the declared `provider:` field is what decides routing."""

    def test_known_providers_route_by_declaration(self):
        for name, provider, wire in [
            ("GPT4.1", "openai", "openai"),
            ("Claude3-Sonnet", "anthropic", "anthropic"),
            ("Gemini2.0-Flash", "google", "google"),
            ("DeepSeek-V3", "deepseek", "openai"),
        ]:
            m = resolve(name)
            self.assertEqual(m.provider, provider, name)
            self.assertEqual(m.wire, wire, name)

    def test_open_weights_never_route_to_an_api(self):
        # `huggingface` is a weights host, not a serving provider.
        m = resolve("Qwen2.5-72B")
        self.assertEqual(m.provider, "vllm")
        self.assertTrue(m.is_open)

    def test_a_name_containing_gpt_is_not_forced_to_openai(self):
        """`gpt-oss-120b` is open weights; substring matching sent it to the
        OpenAI API. Routing must come from the declaration instead."""
        fake = {"gpt-oss-120b": {"type": "open", "provider": "huggingface",
                                 "hf_id": "openai/gpt-oss-120b"}}
        got = alias_for("gpt-oss-120b", "any_bundle", models=fake)
        self.assertEqual(got, "openai/gpt-oss-120b")

    def test_unknown_model_names_what_is_available(self):
        with self.assertRaises(ModelNotFound) as cm:
            resolve("NoSuchModel")
        self.assertIn("GPT4.1", str(cm.exception))


class TestAliasFallback(unittest.TestCase):
    """Convention first: a new bundle must not require editing every model."""

    def test_explicit_alias_wins(self):
        self.assertEqual(alias_for("GPT4.1", "aut_ttcw_cshort"), "gpt_4.1")

    def test_api_model_falls_back_to_api_call(self):
        self.assertEqual(alias_for("GPT4.1", "brand_new_bundle"),
                         "gpt-4.1-2025-04-14")

    def test_open_model_falls_back_to_hf_id(self):
        # These five have no ttct alias at all; the fallback is what makes
        # `--task ttct --model DeepSeek-Qwen-7B` resolvable.
        for name in ["DeepSeek-Qwen-7B", "DeepSeek-Qwen-32B", "DeepSeek-Llama-70B",
                     "OLMo2-13B-SFT", "OLMo2-13B-DPO"]:
            self.assertEqual(alias_for(name, "ttct"), resolve(name).hf_id, name)

    def test_every_registered_model_resolves_for_every_bundle(self):
        """The property that makes adding a bundle free."""
        models = registry.list_models()
        for name in models:
            for bundle in ["aut_ttcw_cshort", "ttct", "neocoder_dat",
                           "math_n_index", "hypothetical_new_bundle"]:
                alias_for(name, bundle, models=models)

    def test_undeciderable_alias_still_raises(self):
        fake = {"Bare": {"type": "api", "provider": "openai"}}
        with self.assertRaises(AliasNotFound):
            alias_for("Bare", "x", models=fake)


class TestTtctAliasesAreFullHfIds(unittest.TestCase):
    """ttct hands the alias straight to vLLM as LLM(model=...), so for an open
    model it must be a full HuggingFace id. Ten of them omitted the org prefix
    until 2026-08-13 and `--task ttct` could not load any open model."""

    def test_open_model_ttct_alias_equals_hf_id(self):
        models = registry.list_models()
        for name, entry in models.items():
            if (entry or {}).get("type") != "open":
                continue
            resolved = alias_for(name, "ttct", models=models)
            self.assertEqual(resolved, entry.get("hf_id"), f"{name} ttct alias")
            self.assertIn("/", resolved, f"{name} ttct alias needs an org prefix")


class TestBaseUrl(unittest.TestCase):
    def test_env_override_points_every_task_at_one_server(self):
        os.environ["CREATIVITYPRISM_MODEL_BASE_URL"] = "http://example:9/v1"
        try:
            self.assertEqual(resolve("Qwen2.5-72B").base_url, "http://example:9/v1")
            self.assertEqual(resolve("GPT4.1").base_url, "http://example:9/v1")
        finally:
            del os.environ["CREATIVITYPRISM_MODEL_BASE_URL"]

    def test_deepseek_keeps_its_endpoint_without_an_override(self):
        self.assertEqual(resolve("DeepSeek-V3").base_url, "https://api.deepseek.com")


class TestClientContract(unittest.TestCase):
    def test_accepts_both_a_string_and_a_messages_list(self):
        from model_hub.client import _as_messages
        self.assertEqual(_as_messages("hi"), [{"role": "user", "content": "hi"}])
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertEqual(_as_messages(msgs), msgs)

    def test_default_concurrency_matches_vllm_max_num_seqs(self):
        """Lowering this does not fail, it silently runs ~4x slower. Measured:
        c=16 is -77.5%, c=64 is -43.4%, c=256 is -1.5% vs offline batch."""
        from model_hub import client
        self.assertEqual(client.DEFAULT_CONCURRENCY, 256)

    def test_batch_can_return_exceptions_instead_of_aborting(self):
        m = ResolvedModel(canonical="x", type="api", provider="openai",
                          model_id="x", api_key="k")
        c = object.__new__(__import__("model_hub").Client)
        c.model, c.extra, c.seed = m, {}, 42
        c.max_tokens, c.temperature, c.top_p, c.top_k = 10, 0, None, None

        def boom(_p, **_k):
            raise RuntimeError("nope")
        c.generate = boom
        out = __import__("model_hub").Client.generate_batch(
            c, ["a", "b"], concurrency=2, return_exceptions=True)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(o, RuntimeError) for o in out))


class TestRawCompletion(unittest.TestCase):
    """Offline `LLM.generate(text)` applies no chat template. A task migrating
    to a server must use complete(), not generate(), or the model sees a
    different prompt and the numbers stop being comparable."""

    def test_openai_wire_exposes_complete(self):
        from model_hub.providers.openai_compat import OpenAICompatProvider
        self.assertTrue(hasattr(OpenAICompatProvider, "complete"))

    def test_providers_without_raw_completion_say_so(self):
        from model_hub.providers.anthropic import AnthropicProvider
        from model_hub.providers.google import GoogleProvider
        for cls in (AnthropicProvider, GoogleProvider):
            self.assertFalse(hasattr(cls, "complete"), cls.__name__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
