"""DAT evaluation: pull ten nouns out of the response, score their spread.

Reimplements ``neocoder_dat/steps/evaluate_dat.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo). No model is loaded --
scoring is GloVe arithmetic -- so the config names none.

The word extraction is lifted verbatim. It is a chain of heuristics tuned against
real outputs (strip the preamble before the first colon, prefer the text after
"Your answer:", prefer bolded words if any are bolded, then split on anything
non-alphabetic), and any "tidying" of it would silently change which words get
scored. Two of its quirks are worth knowing rather than fixing:

* it deletes the literal substrings "user" and "system" anywhere in the text,
  which would also eat them out of the middle of a word;
* it is named ``parse_first_seven`` but collects ten; the cut to seven happens
  later, inside ``Model.dat``.

A response that yields fewer than seven in-vocabulary words scores ``None``, and
the report counts those separately rather than averaging over them.
"""
import os
import re

import numpy as np
import pandas as pd

from src.evaluation.eval_driver import EvalDriver
from src.evaluation.dat_score import Model
from src.utils.helpers import data_path, load_json

DEFAULT_GLOVE_ENV = "CREATIVITYPRISM_GLOVE_PATH"
DEFAULT_WORDS = data_path("dat_words.txt")


def parse_words(text, limit=10):
    """Extract up to `limit` single-word candidates from one response."""
    text = (text or "").strip()

    # Drop everything up to the first colon: models restate the instruction.
    text = re.sub(r"^[^:]*:", "", text)
    text = re.sub(r"user", "", text)
    text = re.sub(r"system", "", text)
    text = re.sub(r"_{5,}", "", text)

    answer_match = re.search(r"Your answer:\s*(.*?)(?:[\n\r]|$)", text)
    if answer_match:
        text = answer_match.group(1)

    word_matches = re.findall(r"\*\*(.*?)\*\*", text)
    if word_matches:
        text = " ".join(word_matches)
    else:
        text = re.sub(r"\*\*", "", text)

    words = []
    for token in re.split(r"[\d\W]+", text):
        token = token.strip()
        if not token or " " in token:
            continue
        words.append(token)
        if len(words) == limit:
            break
    return words


class DATEval(EvalDriver):

    def __init__(self, config={}):
        super().__init__(config)
        self.glove_path = (
            config.get("glove_path")
            or os.environ.get(DEFAULT_GLOVE_ENV)
        )
        if not self.glove_path or not os.path.isfile(self.glove_path):
            raise FileNotFoundError(
                "DAT needs GloVe vectors. Set 'glove_path' in the config or "
                "${} (looked for {!r}). glove.840B.300d.txt is ~5.6 GB "
                "uncompressed.".format(DEFAULT_GLOVE_ENV, self.glove_path)
            )
        self.words_path = config.get("words_path", DEFAULT_WORDS)
        self._model = None

    @property
    def model(self):
        """GloVe is ~5.6 GB of text; load it once, and only if actually scoring."""
        if self._model is None:
            print("=> loading GloVe from {} (this takes a minute)".format(self.glove_path))
            self._model = Model(
                self.glove_path, self.words_path, model_type="glove"
            )
            print("=> {} vectors in vocabulary".format(len(self._model.vectors)))
        return self._model

    def create_batched_prompt(self, input_data):
        """No prompts: one scoring job per generated sample."""
        return [{"row_index": i, "data": item} for i, item in enumerate(input_data)]

    def parse_llm_outputs(self, llm_outputs):
        for result in llm_outputs:
            result["words"] = parse_words(result["data"].get("output"))
        return llm_outputs

    def evaluation(self):
        input_data = load_json(self.config["eval_input_dir"])
        if not input_data:
            raise FileNotFoundError(
                "DAT inference output not found: {}".format(self.config["eval_input_dir"])
            )

        jobs = self.parse_llm_outputs(self.create_batched_prompt(input_data))
        minimum = int(self.config.get("minimum_words", 7))
        print("=> scoring {} sample(s), minimum {} valid words".format(len(jobs), minimum))

        eval_output_cleaned = []
        for job in jobs:
            score = self.model.dat(job["words"], minimum=minimum)
            eval_output_cleaned.append({
                "meta_data": job["data"]["meta_data"],
                "prompt_id": job["data"].get("prompt_id"),
                "output": job["data"].get("output"),
                "words": job["words"],
                # Two numbers, because two things want different ones. `score`
                # is D_0, Olson's mean pairwise cosine distance x100, which the
                # DAT literature quotes. `dat` is the paper's definition,
                # 1 - D_0/100, which is what the published results report.
                # Writing only D_0 here left the report on one scale and the
                # analysis table on the other, pointing opposite ways.
                "score": score,
                "dat": (1 - score / 100) if score is not None else None,
                "scorable": score is not None,
            })

        return self.generate_eval_report(eval_output_cleaned), eval_output_cleaned

    def generate_eval_report(self, eval_outputs):
        """Mean DAT over scorable samples, with the unscorable count beside it.

        Averaging only the scorable ones is the original behaviour, but a model
        that fails to produce seven usable nouns half the time is not comparable
        to one that always does -- hence the count.

        **Direction follows the paper: DAT = 1 - D_0.** The published numbers
        are computed that way, so that is what this reports. ``mean_d0`` keeps
        the raw quantity (Olson's mean pairwise cosine distance x100) alongside
        it, so nothing is lost and the two can be compared directly.
        """
        if not eval_outputs:
            return pd.DataFrame(columns=["n", "n_scored", "n_unscorable",
                                         "mean_dat", "mean_d0", "std_d0"])

        scores = [r["score"] for r in eval_outputs if r["score"] is not None]
        mean_d0 = float(np.mean(scores)) if scores else None
        # The paper's definition, on the 0-1 scale D_0/100 puts it on.
        mean_dat = (1 - mean_d0 / 100) if mean_d0 is not None else None
        return pd.DataFrame([{
            "n": len(eval_outputs),
            "n_scored": len(scores),
            "n_unscorable": len(eval_outputs) - len(scores),
            # DAT = 1 - D_0, as published.
            "mean_dat": mean_dat,
            # The raw quantity behind it: Olson's mean pairwise cosine distance
            # x100, which is what the DAT literature outside this paper quotes.
            "mean_d0": mean_d0,
            "std_d0": float(np.std(scores)) if scores else None,
        }])
