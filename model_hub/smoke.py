"""Pre-flight: can these models actually be run, right now, from this machine.

Every failure this checks for is one that actually happened on 2026-09-03, and
every one of them surfaced *hours into a run* rather than before it started:

* ``generativelanguage.googleapis.com`` does not resolve on Pitt's DNS, so every
  Gemini call died at name lookup -- while ``aiplatform.googleapis.com``, the same
  domain, resolved fine. That is a network fact, not a bad key, and the two are
  told apart here.
* ``claude-3-7-sonnet-20250219`` and ``gemini-2.0-flash`` are retired. The keys
  are valid; the model ids 404. A credentials file keyed by *model name* makes
  this look like "no credentials" instead.
* ``claude-sonnet-4-5`` reasons at length despite a prompt that says "output your
  decision only", so at the paper's 128-token judge budget its verdict is cut off
  before it appears. The old parser scored that as a NO vote.

**A listing is not a check.** ``GET /v1beta/models`` tells you what a provider
advertises, not whether your key may call it, and never whether the answer will
be parseable. Every probe here is a real generation.

Usage::

    python -m model_hub.smoke --models GPT4.1 Qwen2.5-7B
    python -m model_hub.smoke --judges GPT4.1 claude-sonnet-4-5-20250929
    python -m model_hub.smoke --config task_runner/fixtures/my_run.yaml  # models from a run config

Exit code is 0 only when nothing failed. A judge that answers but cannot be
parsed is a failure: it is indistinguishable, downstream, from a judge that
voted no.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass

from .registry import ModelNotFound, list_models, resolve
from .client import Client

# Judges are asked something with exactly one right answer, in the shape the
# real rubrics use: verdict first, explanation after.
JUDGE_PROBE = (
    "Is 2 + 2 equal to 4? Answer YES or NO. Output your decision only."
)
JUDGE_VERDICT = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

# The paper's judge budget is 128 tokens. That is not enough for a current
# reasoning model to reach its verdict, so the probe uses a budget a real run
# should also use, and says so when the verdict does not arrive.
JUDGE_MAX_TOKENS = 1024

MODEL_PROBE = "Reply with the single word OK."
MODEL_MAX_TOKENS = 8

OK = "ok"
NOT_REGISTERED = "not_registered"
NO_CREDENTIALS = "no_credentials"
UNREACHABLE = "unreachable"
AUTH_FAILED = "auth_failed"
MODEL_RETIRED = "model_retired"
RATE_LIMITED = "rate_limited"
PROVIDER_ERROR = "provider_error"
VERDICT_UNPARSEABLE = "verdict_unparseable"
SKIPPED_OPEN = "skipped_open"
ENDPOINT_DOWN = "endpoint_down"

FAILING = {
    NOT_REGISTERED, NO_CREDENTIALS, UNREACHABLE, AUTH_FAILED,
    MODEL_RETIRED, RATE_LIMITED, PROVIDER_ERROR, VERDICT_UNPARSEABLE,
    ENDPOINT_DOWN,
}

# Ordered: the first pattern that matches wins, so the specific ones come first.
_SIGNALS = (
    (UNREACHABLE, (
        "name or service not known", "nodename nor servname", "getaddrinfo",
        "temporary failure in name resolution", "failed to resolve",
        "connection refused", "connection reset", "connection aborted",
        "connection error", "max retries exceeded", "cannot connect",
        "network is unreachable", "timed out", "timeout",
    )),
    (MODEL_RETIRED, (
        "no longer available", "not_found", "not found", "does not exist",
        "is not supported", "404", "model_not_found", "deprecated",
    )),
    (AUTH_FAILED, (
        "401", "403", "unauthorized", "invalid_api_key", "invalid api key",
        "permission denied", "authentication", "unregistered callers",
    )),
    (RATE_LIMITED, ("429", "rate limit", "quota", "resource_exhausted")),
)


def classify(message):
    """Map a provider error to one of the known failure modes.

    Text matching, because every provider words its failures differently and
    none of them expose a stable machine-readable kind. Anything unrecognised
    stays ``provider_error`` rather than being forced into a bucket -- a wrong
    label here would send someone to fix the wrong thing.
    """
    low = (message or "").lower()
    for status, needles in _SIGNALS:
        if any(n in low for n in needles):
            return status
    return PROVIDER_ERROR


@dataclass
class Probe:
    canonical: str
    role: str                 # "model" | "judge"
    status: str
    detail: str = ""
    latency_s: float | None = None
    reply: str = ""

    @property
    def ok(self):
        return self.status not in FAILING

    def line(self):
        mark = {OK: "PASS", SKIPPED_OPEN: "SKIP"}.get(self.status, "FAIL")
        timing = "" if self.latency_s is None else f" {self.latency_s:.1f}s"
        detail = f" -- {self.detail}" if self.detail else ""
        return f"  {mark}: [{self.role}] {self.canonical} ({self.status}{timing}){detail}"


def _probe(canonical, role, bundle=None, keys_path=None, max_tokens=None,
           prompt=None, require_verdict=False, timeout_note=""):
    try:
        resolved = resolve(canonical, bundle=bundle, keys_path=keys_path)
    except ModelNotFound as e:
        return Probe(canonical, role, NOT_REGISTERED, str(e))
    except Exception as e:                                  # noqa: BLE001
        return Probe(canonical, role, PROVIDER_ERROR, str(e))

    if resolved.is_open and not resolved.base_url:
        return Probe(
            canonical, role, SKIPPED_OPEN,
            "open-weight model with no endpoint; set CREATIVITYPRISM_MODEL_BASE_URL "
            "to check a running server, otherwise it is only checkable by loading it",
        )

    if resolved.provider != "vllm" and not resolved.api_key:
        return Probe(
            canonical, role, NO_CREDENTIALS,
            f"no key for provider {resolved.provider!r} / model {resolved.model_id!r}. "
            "The credentials file is keyed by model name, so a renamed or replacement "
            "model needs its own entry.",
        )

    started = time.time()
    try:
        client = Client(resolved, max_tokens=max_tokens, temperature=0)
        reply = client.generate(prompt)
    except Exception as e:                                  # noqa: BLE001 - classified, not swallowed
        elapsed = round(time.time() - started, 2)
        status = classify(str(e))
        # An open-weight model is reached through a server someone has to start.
        # "unreachable" is true but useless; name the URL and what to do.
        if resolved.is_open and status == UNREACHABLE:
            return Probe(
                canonical, role, ENDPOINT_DOWN,
                f"no vLLM server answering at {resolved.base_url}. Start one for "
                f"{resolved.model_id}, or run this model offline -- an open-weight "
                "model with no endpoint cannot be checked without loading it.",
                elapsed,
            )
        return Probe(canonical, role, status, str(e)[:200], elapsed)
    elapsed = round(time.time() - started, 2)

    if not isinstance(reply, str) or not reply.strip():
        return Probe(canonical, role, PROVIDER_ERROR,
                     "empty reply", elapsed, reply or "")

    if require_verdict and not JUDGE_VERDICT.search(reply):
        return Probe(
            canonical, role, VERDICT_UNPARSEABLE,
            f"answered {len(reply)} chars with no YES/NO in them{timeout_note}. "
            "Downstream this is indistinguishable from a NO vote.",
            elapsed, reply[:200],
        )

    return Probe(canonical, role, OK, "", elapsed, reply.strip()[:80])


def check_model(canonical, bundle=None, keys_path=None):
    """Can this model be asked for one token and answer."""
    return _probe(canonical, "model", bundle, keys_path,
                  max_tokens=MODEL_MAX_TOKENS, prompt=MODEL_PROBE)


def check_judge(canonical, bundle=None, keys_path=None, max_tokens=JUDGE_MAX_TOKENS):
    """Can this model be asked for a verdict and give a parseable one."""
    return _probe(
        canonical, "judge", bundle, keys_path, max_tokens=max_tokens,
        prompt=JUDGE_PROBE, require_verdict=True,
        timeout_note=f" at max_tokens={max_tokens}",
    )


def run(models=(), judges=(), bundle=None, keys_path=None,
        judge_max_tokens=JUDGE_MAX_TOKENS):
    probes = [check_model(m, bundle, keys_path) for m in models]
    probes += [check_judge(j, bundle, keys_path, judge_max_tokens) for j in judges]
    return probes


def report(probes, stream=sys.stdout):
    for probe in probes:
        print(probe.line(), file=stream)
    failed = [p for p in probes if not p.ok]
    skipped = [p for p in probes if p.status == SKIPPED_OPEN]
    print(
        f"\n=== {len(probes) - len(failed) - len(skipped)} usable, "
        f"{len(failed)} failed, {len(skipped)} unchecked ===",
        file=stream,
    )
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check that models and judges can actually be run before a run starts."
    )
    parser.add_argument("--models", nargs="*", default=[],
                        help="Canonical names to probe with a one-token generation.")
    parser.add_argument("--judges", nargs="*", default=[],
                        help="Canonical names to probe with a verdict question; a "
                             "reply with no YES/NO in it counts as a failure.")
    parser.add_argument("--bundle", default=None,
                        help="Resolve aliases as this bundle would (aut, ttct, ...).")
    parser.add_argument("--keys-path", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--list", action="store_true",
                        help="Print the registered model names and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(list_models()):
            print(name)
        return 0

    if not args.models and not args.judges:
        parser.error("give --models and/or --judges (or --list)")

    probes = run(args.models, args.judges, args.bundle, args.keys_path,
                 args.judge_max_tokens)
    return report(probes)


if __name__ == "__main__":
    raise SystemExit(main())
