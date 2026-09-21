"""Creativity Index evaluation: DJ Search against an n-gram index.

Ported from ``math_n_index/src/evaluation/evaluation_creative_index_*.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo),
of which there were two copies that disagreed. Reconciled here, with the
disagreements named rather than silently resolved:

* **They queried different corpora.** ``..._exact.py`` used ``v4_rpj_llama_s4``
  and ``..._parr.py`` used ``v4_dolma-v1_7_llama``. These are not interchangeable:
  the same phrase counts 457,312 in one and 915,616 in the other, so the two
  scripts produced creativity indices that cannot be compared. The corpus is now
  a required config field with no default -- picking one silently is exactly how
  this went unnoticed. **Resolved 2026-09-03: the published runs used
  ``v4_dolma-v1_7_llama``.** Every sbatch script in ``math_n_index`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo)
  (book/poem/speech) invokes ``evaluation_creative_index_parr``, labels the run
  ``--task <model>-<domain>-dolma``, and one writes to ``data/evaluation_dolma/``;
  the README documents the same command, and ``parr`` is the only one of the two
  that ever received a bug fix. ``..._exact.py`` and its rpj corpus appear to be
  dead.
* **Only ``..._parr.py`` handled a failed lookup.** ``..._exact.py`` scored an API
  error as ``occurrence = 0``, identical to "this n-gram is not in the corpus".
  That *lowers* coverage, which *raises* the reported creativity index -- a run
  that quietly loses lookups publishes a flattering number. The fail-loud
  behaviour is kept.
* **The gated tokenizer no longer blocks the import**, and this is load-bearing
  rather than cosmetic: every production script passes ``--lm_tokenizer``, so the
  Llama tokenizer path -- not the NLTK default -- is the one that produced the
  published numbers. Both modules built
  ``AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")`` at module scope,
  so the gated repo stopped the file from importing at all -- even on the default
  path, which uses the NLTK tokenizer and never touches it. It is now loaded
  lazily, from an ungated mirror whose token ids were checked against the ids
  infini-gram reports for the same strings.

No model is loaded: scoring is an HTTP lookup, so the config names no model.

Score. Coverage is the fraction of a response covered by spans that appear in the
corpus, so L-uniqueness is ``1 - coverage``. The paper sums L-uniqueness over
n-gram sizes 5 through 12 inclusive and divides by 8, which is the largest the sum
can reach.
"""
import concurrent.futures
import random
import time

import nltk
import numpy as np
import pandas as pd
import requests
from unidecode import unidecode

from src.evaluation.eval_driver import EvalDriver
from src.utils.helpers import load_json

API_URL = "https://api.infini-gram.io/"

# A count normally answers in ~0.12s regardless of span length. These caps stop a
# single hung request from taking a worker out of service for minutes; see
# _count_ngram for the measurements behind them.
REQUEST_TIMEOUT = 15.0
MAX_BACKOFF = 8.0

# The paper: "We sum over the L-Uniqueness with spans of n-grams from 5 to 12
# inclusively ... then we normalize the score by dividing it with 8".
NGRAM_SIZES = tuple(range(5, 13))
NGRAM_NORMALISER = float(len(NGRAM_SIZES))

# An ungated mirror of the Llama-2 tokenizer. Only used with lm_tokenizer=True;
# verified to produce the same token ids that infini-gram reports for the same
# strings, including the paper's own example prompts.
LM_TOKENIZER_ID = "NousResearch/Llama-2-7b-hf"


class LookupFailed(RuntimeError):
    """An n-gram count could not be obtained. Never treat this as a zero count."""


def _count_ngram(span_text, corpus, max_retries=8, base_delay=0.5, request_delay=0.1,
                 timeout=REQUEST_TIMEOUT, max_backoff=MAX_BACKOFF):
    """One infini-gram count, or raise.

    The public endpoint rate-limits by IP and answers a plain 403 while doing so.
    Eight workers is already too many: measured 2026-09-03, eight in flight drove
    it to refuse 10 out of 10 probe requests, and the refusal outlasted the run.
    Retries back off exponentially with jitter so parallel workers do not retry in
    lockstep, but backoff cannot buy capacity that is not there -- keep
    ``num_workers`` low instead.

    Both the timeout and the backoff are capped, and that is load-bearing rather
    than tidiness. A healthy count answers in ~0.12s (measured 2026-09-04, 60/60
    at 7.3 req/s), but the endpoint intermittently hangs one. With a 60s timeout
    and uncapped doubling, a single unlucky span could occupy a worker for
    8 x 60s + ~64s of sleeps, about nine minutes -- and with two or three workers
    that is a third to a half of all throughput, spent on one n-gram. The run
    then looks hung rather than slow, which is how a sweep that was working got
    read as an endpoint that was refusing us.
    """
    request_data = {
        "corpus": corpus,
        "engine": "c++",
        "query_type": "count",
        "query": span_text,
    }
    search_result = None
    for attempt in range(max_retries):
        try:
            time.sleep(request_delay)
            response = requests.post(API_URL, json=request_data, timeout=timeout)
            response.raise_for_status()
            search_result = response.json()
            break
        except (requests.RequestException, ValueError):
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_backoff)
                time.sleep(delay + random.uniform(0, base_delay))

    if search_result is None or "count" not in search_result:
        raise LookupFailed(
            "infini-gram lookup failed after {} attempts for span {!r} on corpus {!r}. "
            "Refusing to score: a missing lookup is indistinguishable from a zero "
            "count and would inflate the creativity index. If this fired early in a "
            "run, it is rate limiting, not a bad span -- lower 'num_workers' and "
            "raise 'request_delay'.".format(max_retries, span_text, corpus)
        )
    return search_result["count"]


def find_exact_match(detokenize, tokens, min_ngram, corpus, request_delay=0.1,
                     on_request=None, timeout=REQUEST_TIMEOUT, max_backoff=MAX_BACKOFF):
    """Greedy longest-match sweep; returns the fraction of tokens covered.

    Walks a window over the response, growing it while the span still occurs in
    the corpus and sliding it when it does not. Identical in shape to the
    original, with the span bookkeeping reduced to the coverage flags it existed
    to produce.
    """
    covered = [False] * len(tokens)
    first, second = 0, min_ngram
    last_end = 0

    requests_made = 0
    while second <= len(tokens):
        span_text = detokenize(tokens[first:second])
        requests_made += 1
        if on_request is not None:
            on_request(requests_made)
        if _count_ngram(span_text, corpus, request_delay=request_delay,
                        timeout=timeout, max_backoff=max_backoff):
            covered[first:second] = [True] * (second - first)
            last_end = max(last_end, second)
            second += 1
        elif second - first > min_ngram:
            first += 1
        else:
            first += 1
            second += 1

    # The original measured coverage over the whole document once the sweep ran
    # to the end, which it always does here.
    return sum(covered) / len(covered) if covered else 0.0


class CreativityIndexEval(EvalDriver):

    def __init__(self, config={}):
        super().__init__(config)
        self.corpus = config.get("corpus")
        if not self.corpus:
            raise ValueError(
                "creativity_index needs an explicit 'corpus'. The two original "
                "scripts disagreed (v4_rpj_llama_s4 vs v4_dolma-v1_7_llama) and "
                "their scores are not comparable, so there is no safe default."
            )
        self._tokenizer = None

    def _tokenize_funcs(self):
        """NLTK casual tokenizer by default; the LM tokenizer only when asked."""
        if not self.config.get("lm_tokenizer", False):
            from sacremoses import MosesDetokenizer
            md = MosesDetokenizer(lang="en")
            return (nltk.tokenize.casual.casual_tokenize, md.detokenize)

        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                LM_TOKENIZER_ID, add_bos_token=False, add_eos_token=False
            )
        tok = self._tokenizer
        return (
            tok.tokenize,
            lambda x: tok.decode(tok.convert_tokens_to_ids(x)),
        )

    def create_batched_prompt(self, input_data, min_ngram=None):
        """The unit of work is (document, n-gram size), not a prompt.

        Kept on the contract's name because the pipeline calls it, but there is
        no LLM prompt here: each entry is one DJ Search to run.
        """
        sizes = self.config.get("ngram_sizes", list(NGRAM_SIZES)) if min_ngram is None else [min_ngram]
        jobs = []
        for item in input_data:
            response = item.get("response")
            if not isinstance(response, str) or not response.strip():
                continue
            for size in sizes:
                jobs.append({
                    "prompt_id": item.get("prompt_id") or item["meta_data"]["id"],
                    "domain": item.get("domain"),
                    "min_ngram": size,
                    "data": item,
                })
        return jobs

    def _run_job(self, job):
        tokenize, detokenize = self._tokenize_funcs()
        tokens = tokenize(unidecode(job["data"]["response"]))
        if len(tokens) <= job["min_ngram"]:
            return None
        # A sweep is hundreds of round trips. Without a counter there is no way
        # to tell a slow run from a hung one until the whole sweep lands.
        import threading, time as _t
        state = {"n": 0, "t0": _t.time()}
        every = int(self.config.get("progress_every", 100))
        lock = getattr(self, "_print_lock", None)
        if lock is None:
            lock = self._print_lock = threading.Lock()

        def tick(n):
            state["n"] = n
            if every and n % every == 0:
                rate = n / max(1e-6, _t.time() - state["t0"])
                with lock:
                    print("      {} n={} .. {} req, {:.1f}/s".format(
                        job["prompt_id"], job["min_ngram"], n, rate), flush=True)

        coverage = find_exact_match(
            detokenize, tokens, job["min_ngram"], self.corpus,
            request_delay=float(self.config.get("request_delay", 0.1)),
            on_request=tick,
            timeout=float(self.config.get("request_timeout", REQUEST_TIMEOUT)),
            max_backoff=float(self.config.get("max_backoff", MAX_BACKOFF)),
        )
        return dict(job, coverage=coverage, n_tokens=len(tokens))

    def parse_llm_outputs(self, llm_outputs):
        """No LLM output to parse; coverage arrives as a float."""
        return llm_outputs

    def _checkpoint_path(self):
        return self.config.get(
            "checkpoint_path",
            "data/output/{}/eval_checkpoint.json".format(self.config["run_id"]),
        )

    def _load_checkpoint(self):
        """Sweeps already done, keyed by (prompt_id, min_ngram).

        A full sweep is hours of HTTP and one throttle can end it, so finishing
        work must survive the process. The original had this and the first port
        dropped it; a 13-hour run that loses everything on the last lookup is
        not a run.
        """
        import os
        path = self._checkpoint_path()
        if not os.path.isfile(path):
            return {}
        try:
            done = load_json(path)
        except Exception:                                   # noqa: BLE001
            return {}
        return {(r["prompt_id"], r["min_ngram"]): r for r in done
                if r.get("corpus") == self.corpus}

    def _save_checkpoint(self, rows):
        import json as _json
        import os
        path = self._checkpoint_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            _json.dump(rows, fh, indent=2)
        os.replace(tmp, path)                               # atomic; a killed run
                                                            # never leaves half a file

    def evaluation(self):
        input_data = load_json(self.config["eval_input_dir"])
        if not input_data:
            raise FileNotFoundError(
                "creativity_index inference output not found: {}".format(
                    self.config["eval_input_dir"]
                )
            )

        jobs = self.create_batched_prompt(input_data)
        workers = int(self.config.get("num_workers", 2))

        # Resume: a sweep already recorded for this corpus is not repeated.
        done = self._load_checkpoint()
        pending = [j for j in jobs if (j["prompt_id"], j["min_ngram"]) not in done]
        print("=> {} sweep(s) over corpus {} with {} worker(s)"
              "{}".format(len(jobs), self.corpus, workers,
                          "; {} already done, {} to go".format(
                              len(jobs) - len(pending), len(pending))
                          if len(pending) != len(jobs) else ""))
        jobs = pending

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._run_job, job) for job in jobs]
            for finished, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                # A LookupFailed here propagates and ends the run, by design --
                # but the checkpoint below means the sweeps already finished are
                # kept, so re-running picks up where this stopped.
                result = fut.result()
                if result is not None:
                    results.append(result)
                    self._save_checkpoint(
                        list(done.values()) + [self._row(r) for r in results])
                # A sweep is hundreds of HTTP round trips, so a run with no
                # progress line is indistinguishable from a hung one.
                print("   [{}/{}] {} n={} coverage={:.3f}".format(
                    finished, len(jobs),
                    (result or {}).get("prompt_id", "-"),
                    (result or {}).get("min_ngram", "-"),
                    (result or {}).get("coverage", float("nan")),
                ), flush=True)

        eval_output_cleaned = list(done.values()) + [self._row(r) for r in results]
        return self.generate_eval_report(eval_output_cleaned), eval_output_cleaned

    def _row(self, r):
        return {
            "meta_data": r["data"]["meta_data"],
            "prompt_id": r["prompt_id"],
            "domain": r["domain"],
            "prompt": r["data"].get("prompt"),
            "response": r["data"]["response"],
            "min_ngram": r["min_ngram"],
            "coverage": r["coverage"],
            "n_tokens": r["n_tokens"],
            "corpus": self.corpus,
        }

    def generate_eval_report(self, eval_outputs):
        """Per document: L-uniqueness at each size, and the normalised index.

        A document is only given an index once every size in the sweep scored it;
        summing a partial sweep would understate it, and reporting that number
        next to a complete one would be worse than reporting nothing.
        """
        if not eval_outputs:
            return pd.DataFrame(columns=[
                "prompt_id", "domain", "n_sizes", "creativity_index", "complete"])

        df = pd.DataFrame(eval_outputs)
        df["l_uniqueness"] = 1.0 - df["coverage"]

        rows = []
        for (prompt_id, domain), group in df.groupby(["prompt_id", "domain"], dropna=False):
            complete = len(group) == len(NGRAM_SIZES)
            rows.append({
                "prompt_id": prompt_id,
                "domain": domain,
                "n_sizes": len(group),
                "creativity_index": (
                    group["l_uniqueness"].sum() / NGRAM_NORMALISER if complete else np.nan
                ),
                "complete": complete,
            })
        return pd.DataFrame(rows).sort_values(["domain", "prompt_id"]).reset_index(drop=True)
