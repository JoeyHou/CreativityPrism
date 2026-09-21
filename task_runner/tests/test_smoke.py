"""Gate for the pre-flight checker. No network: it tests the classification.

The point of `model_hub.smoke` is that the four ways a model can be unusable get
four different answers, because they need four different fixes. A checker that
called everything "provider_error" would be worse than none at all -- it would
send someone to rotate a key when the real problem was DNS. These cases are the
real strings the providers returned on 2026-09-03.
"""
import unittest

from model_hub import smoke


class TestClassify(unittest.TestCase):

    def test_dns_failure_is_unreachable_not_auth(self):
        # Pitt's resolver withholds generativelanguage.googleapis.com. Reading
        # this as an auth problem is what sent us looking at the API key.
        self.assertEqual(
            smoke.classify("google/gemini-2.0-flash: [Errno -2] Name or service not known"),
            smoke.UNREACHABLE,
        )

    def test_retired_model_is_not_missing_credentials(self):
        # The key is valid; the model id is gone. Because the credentials file is
        # keyed by model name, this looks like "no key" unless it is separated.
        self.assertEqual(
            smoke.classify(
                "anthropic/claude-3-7-sonnet-20250219: Error code: 404 - "
                "{'type': 'not_found_error', 'message': 'model: claude-3-7-sonnet-20250219'}"
            ),
            smoke.MODEL_RETIRED,
        )
        self.assertEqual(
            smoke.classify("This model models/gemini-2.0-flash is no longer available."),
            smoke.MODEL_RETIRED,
        )

    def test_auth_failure(self):
        self.assertEqual(
            smoke.classify("Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}"),
            smoke.AUTH_FAILED,
        )
        self.assertEqual(
            smoke.classify("Method doesn't allow unregistered callers"),
            smoke.AUTH_FAILED,
        )

    def test_rate_limit(self):
        self.assertEqual(smoke.classify("Error code: 429 - rate limit exceeded"),
                         smoke.RATE_LIMITED)

    def test_dead_endpoint_reads_as_unreachable(self):
        # openai-python says only "Connection error." when nothing is listening.
        self.assertEqual(smoke.classify("vllm/Qwen/Qwen2.5-7B-Instruct: Connection error."),
                         smoke.UNREACHABLE)

    def test_unknown_stays_unknown(self):
        # Never guess. A wrong label costs more than an honest "don't know".
        self.assertEqual(smoke.classify("something nobody has seen before"),
                         smoke.PROVIDER_ERROR)
        self.assertEqual(smoke.classify(""), smoke.PROVIDER_ERROR)


class TestVerdictParsing(unittest.TestCase):

    def test_verdict_found_whole_word_only(self):
        self.assertTrue(smoke.JUDGE_VERDICT.search("YES"))
        self.assertTrue(smoke.JUDGE_VERDICT.search("The answer is no, because ..."))
        # "YESTERDAY" used to match a substring test and score as a YES vote.
        self.assertIsNone(smoke.JUDGE_VERDICT.search("YESTERDAY"))

    def test_reasoning_without_a_verdict_is_a_failure(self):
        # claude-sonnet-4-5 at the paper's 128-token budget: pages of analysis,
        # no verdict. The old parser turned this into a NO vote.
        truncated = ("To determine if the new solution is novel, I need to compare "
                     "it against the reference solution using the given criteria.")
        self.assertIsNone(smoke.JUDGE_VERDICT.search(truncated))

    def test_judge_budget_is_not_the_papers_128(self):
        # 128 truncates a current model before its verdict; measured 1024 is enough.
        self.assertGreaterEqual(smoke.JUDGE_MAX_TOKENS, 1024)


class TestProbeStatus(unittest.TestCase):

    def test_every_failure_mode_counts_as_failing(self):
        for status in (smoke.NOT_REGISTERED, smoke.NO_CREDENTIALS, smoke.UNREACHABLE,
                       smoke.AUTH_FAILED, smoke.MODEL_RETIRED, smoke.RATE_LIMITED,
                       smoke.PROVIDER_ERROR, smoke.VERDICT_UNPARSEABLE,
                       smoke.ENDPOINT_DOWN):
            self.assertFalse(smoke.Probe("m", "model", status).ok, status)

    def test_ok_and_skip_are_not_failures(self):
        self.assertTrue(smoke.Probe("m", "model", smoke.OK).ok)
        self.assertTrue(smoke.Probe("m", "model", smoke.SKIPPED_OPEN).ok)

    def test_unregistered_model_needs_no_network(self):
        probe = smoke.check_model("DefinitelyNotARegisteredModel")
        self.assertEqual(probe.status, smoke.NOT_REGISTERED)
        # The message has to name the alternatives, or the next step is a grep.
        self.assertIn("GPT4.1", probe.detail)


if __name__ == "__main__":
    unittest.main()
