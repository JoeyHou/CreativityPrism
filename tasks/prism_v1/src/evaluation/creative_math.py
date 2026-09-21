"""Creative Math evaluation: correctness, then coarse-grained novelty.

Ported from ``math_n_index/src/evaluation/creative_math_evaluation.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo) and
``creative_math_eval_api.py``, which disagreed with each other *and* with the
paper. Every disagreement is surfaced as an explicit config choice rather than
resolved silently, because each one moves a published number:

* **Correctness rule.** The paper says "We use Claude3-Sonnet as the sole
  correctness evaluator", and explains that the three-model ensemble of the
  original CreativeMath work was dropped. Both code paths instead require *all*
  judges to say YES. Unanimity of three is strictly harsher than one judge, so
  the two rules cannot produce the same correctness ratio. ``correctness_rule``
  is required, with no default: ``"single"`` follows the paper, ``"unanimous"``
  follows the code.
* **Novelty tie-break.** ``creative_math_evaluation.py`` used ``yes > no``;
  ``creative_math_eval_api.py`` used ``yes >= no``. With three live judges there
  are no ties, so this only bites when a judge fails -- exactly when the two
  files then disagree about the answer. ``"strict"`` here, matching the stricter
  of the two.
* **Fine-grained novelty.** The paper says it was not conducted, and gives its
  reasons. Both code paths run it. Off unless ``fine_grained`` is set.
* **An unreadable verdict is not a NO.** ``extract_yes_no`` returned "NO" when it
  found neither word, so an API error, a refusal, or a truncated reply all
  scored as "the solution is wrong". Here it returns ``None`` and the row records
  which judges failed, so a broken judge looks like a broken judge.

The paper also cleans generations with Llama-3.3-70B-Instruct before judging.
That model needs four GPUs, so it is a separate stage; every row carries the
``cleaned`` flag it arrived with, and the report says how many rows were judged
uncleaned. A run of uncleaned generations is not comparable to the paper.
"""
import re

import pandas as pd

from src.evaluation.eval_driver import EvalDriver
from src.utils.helpers import load_json
from src.prompt_engineering.creative_math_prompts import (
    load_coarse_grained_novelty_evaluation_prompt,
    load_correctness_evaluation_prompt,
    load_fine_grained_novelty_evaluation_prompt,
)
from src.utils.api_wrapper import ModelWrapper

_YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

# Any string a provider wrapper returns instead of raising. ModelWrapper converts
# exceptions to text like "OpenAI API Error: ...", which would otherwise be parsed
# as a verdict.
_ERROR_MARKERS = ("API Error:", "api error:")


def extract_yes_no(response):
    """First whole-word verdict, or None when there is none.

    Whole words only: a substring test fired on "YESTERDAY". First match wins,
    because every prompt asks for the verdict before the explanation. Returning
    None rather than "NO" is the difference between "the judge said no" and "the
    judge did not answer".
    """
    if not isinstance(response, str) or any(m in response for m in _ERROR_MARKERS):
        return None
    match = _YES_NO_RE.search(response)
    return match.group(1).upper() if match else None


class CreativeMathEval(EvalDriver):

    def __init__(self, config={}):
        super().__init__(config)

        self.judges = config.get("judges") or []
        if not self.judges:
            raise ValueError("creative_math needs at least one judge in 'judges'.")

        self.correctness_rule = config.get("correctness_rule")
        if self.correctness_rule not in ("single", "unanimous"):
            raise ValueError(
                "creative_math needs an explicit 'correctness_rule'. The paper says "
                "one judge decides correctness ('single'); both code paths required "
                "every judge to agree ('unanimous'). They give different correctness "
                "ratios, so there is no safe default."
            )
        self.correctness_judge = config.get("correctness_judge") or self.judges[0]
        if self.correctness_rule == "single" and self.correctness_judge not in self.judges:
            raise ValueError(
                "correctness_judge {!r} is not in judges {}".format(
                    self.correctness_judge, self.judges
                )
            )

        self.fine_grained = bool(config.get("fine_grained", False))

        # One wrapper per judge. Driver's own self.llm is unused here: the judges
        # are named per stage, not once for the run.
        # The credentials file is keyed by *model* name, so a provider's other
        # models have no entry even though the same key opens them -- and model
        # names retire (claude-3-7-sonnet-20250219 now 404s on a working key).
        # judge_credentials maps a judge to the entry whose key it should use.
        keys = self.api_models
        cred_map = config.get("judge_credentials", {})
        resolved = {j: cred_map.get(j, j) for j in self.judges}
        missing = [j for j, c in resolved.items() if c not in keys]
        if missing:
            raise ValueError(
                "no credentials for judge(s) {}. Either add an entry, or map it with "
                "judge_credentials: {{judge: existing_entry}}. Entries present: {}".format(
                    missing, sorted(keys)
                )
            )
        self._judges = {j: ModelWrapper(j, keys[c]) for j, c in resolved.items()}
        self._judge_config = {
            "temperature": config.get("judge_temperature", 0.0),
            "max_tokens": config.get("judge_max_tokens", 128),
        }

    def _ask(self, judge, prompt):
        response = self._judges[judge].generate_response(
            [{"role": "user", "content": prompt}], self._judge_config
        )
        return response, extract_yes_no(response)

    def create_batched_prompt(self, input_data, stage="correctness"):
        """One entry per (row, judge) for the given stage."""
        jobs = []
        for index, item in enumerate(input_data):
            problem = item["problem"]
            solutions = item["all_solutions"]
            new_solution = item["response"]
            for judge in self.judges:
                if stage == "correctness":
                    prompt = load_correctness_evaluation_prompt(
                        problem, solutions, new_solution)
                elif stage == "coarse":
                    prompt = load_coarse_grained_novelty_evaluation_prompt(
                        problem, solutions, item["k"], new_solution)
                else:
                    prompt = load_fine_grained_novelty_evaluation_prompt(
                        problem, solutions, item["k"], new_solution)
                jobs.append({
                    "row_index": index, "judge": judge,
                    "stage": stage, "prompt_text": prompt,
                })
        return jobs

    def parse_llm_outputs(self, llm_outputs):
        for result in llm_outputs:
            result["decision"] = extract_yes_no(result.get("raw_output"))
        return llm_outputs

    def _run_stage(self, input_data, stage, only_rows=None):
        """Ask every judge about every eligible row; return {row: {judge: verdict}}."""
        verdicts = {}
        jobs = self.create_batched_prompt(input_data, stage)
        for job in jobs:
            if only_rows is not None and job["row_index"] not in only_rows:
                continue
            response, decision = self._ask(job["judge"], job["prompt_text"])
            verdicts.setdefault(job["row_index"], {})[job["judge"]] = {
                "decision": decision, "response": response,
            }
        return verdicts

    @staticmethod
    def _decisions(row_verdicts):
        return [v["decision"] for v in row_verdicts.values() if v["decision"] is not None]

    def _correctness_final(self, row_verdicts):
        if self.correctness_rule == "single":
            verdict = row_verdicts.get(self.correctness_judge, {}).get("decision")
            return verdict, (verdict is None)
        decisions = self._decisions(row_verdicts)
        # A judge that did not answer must not be counted as agreement, and must
        # not be counted as dissent either -- the row is simply unresolved.
        if len(decisions) != len(self.judges):
            return None, True
        return ("YES" if all(d == "YES" for d in decisions) else "NO"), False

    def _majority(self, row_verdicts):
        decisions = self._decisions(row_verdicts)
        if not decisions:
            return None, True
        yes = sum(1 for d in decisions if d == "YES")
        no = len(decisions) - yes
        # Strict majority: the two originals disagreed on ties and ties only ever
        # happen when a judge has already failed.
        return ("YES" if yes > no else "NO"), (len(decisions) != len(self.judges))

    def evaluation(self):
        input_data = load_json(self.config["eval_input_dir"])
        if not input_data:
            raise FileNotFoundError(
                "creative_math inference output not found: {}".format(
                    self.config["eval_input_dir"]))

        test_size = self.config.get("test_size", -1)
        if test_size and test_size > 0:
            input_data = input_data[:int(test_size)]

        print("=> correctness ({} rule) over {} row(s) x {} judge(s)".format(
            self.correctness_rule, len(input_data), len(self.judges)))
        correctness = self._run_stage(input_data, "correctness")

        correct_rows = set()
        for index in range(len(input_data)):
            final, _ = self._correctness_final(correctness.get(index, {}))
            if final == "YES":
                correct_rows.add(index)

        # Novelty is only asked about solutions that are correct: an incorrect
        # solution being "different" is not creativity.
        print("=> coarse novelty over {} correct row(s)".format(len(correct_rows)))
        coarse = self._run_stage(input_data, "coarse", only_rows=correct_rows)

        fine = {}
        if self.fine_grained:
            novel_rows = {i for i in correct_rows
                          if self._majority(coarse.get(i, {}))[0] == "YES"
                          and input_data[i]["k"] != input_data[i]["n"]}
            print("=> fine novelty over {} row(s)".format(len(novel_rows)))
            fine = self._run_stage(input_data, "fine", only_rows=novel_rows)

        eval_output_cleaned = []
        for index, item in enumerate(input_data):
            c_final, c_incomplete = self._correctness_final(correctness.get(index, {}))
            row = {
                "meta_data": item["meta_data"],
                "problem_id": item["problem_id"],
                # question_number is part of a row's identity, not decoration:
                # the same problem_id appears once per question and once per k,
                # so dropping it here left the analysis loader joining
                # (problem_id, None, k) against the inference side's
                # (problem_id, 0, k) -- no match, and every judged verdict was
                # reported as unscored even though it was sitting in this file.
                "question_number": item.get("question_number"),
                "k": item["k"],
                "n": item["n"],
                "response": item["response"],
                "cleaned": item.get("cleaned", False),
                "correctness": {j: v["decision"] for j, v in correctness.get(index, {}).items()},
                "correctness_final": c_final,
                "correctness_incomplete": c_incomplete,
                "correctness_rule": self.correctness_rule,
                "judges": list(self.judges),
            }
            if index in correct_rows:
                n_final, n_incomplete = self._majority(coarse.get(index, {}))
                row["coarse_grained_novelty"] = {
                    j: v["decision"] for j, v in coarse.get(index, {}).items()}
                row["coarse_grained_novelty_final"] = n_final
                row["coarse_grained_novelty_incomplete"] = n_incomplete
            else:
                row["coarse_grained_novelty"] = {}
                row["coarse_grained_novelty_final"] = None
                row["coarse_grained_novelty_incomplete"] = False
            if self.fine_grained and index in fine:
                f_final, f_incomplete = self._majority(fine.get(index, {}))
                row["fine_grained_novelty_final"] = f_final
                row["fine_grained_novelty_incomplete"] = f_incomplete
            eval_output_cleaned.append(row)

        return self.generate_eval_report(eval_output_cleaned), eval_output_cleaned

    def generate_eval_report(self, eval_outputs):
        """Ratios, plus the counts that say whether they can be trusted.

        ``*_unresolved`` is the number of rows where at least one judge gave no
        readable verdict. A ratio computed over rows that include unresolved ones
        is not the ratio the paper reports, and the only way to notice is to
        print the count next to it.
        """
        if not eval_outputs:
            return pd.DataFrame(columns=["metric", "yes", "total", "ratio", "unresolved"])

        total = len(eval_outputs)
        c_yes = sum(1 for r in eval_outputs if r["correctness_final"] == "YES")
        c_unres = sum(1 for r in eval_outputs if r["correctness_incomplete"])

        novelty_rows = [r for r in eval_outputs if r["coarse_grained_novelty_final"] is not None]
        n_yes = sum(1 for r in novelty_rows if r["coarse_grained_novelty_final"] == "YES")
        n_unres = sum(1 for r in eval_outputs if r.get("coarse_grained_novelty_incomplete"))

        uncleaned = sum(1 for r in eval_outputs if not r.get("cleaned", False))

        rows = [
            {"metric": "correctness", "yes": c_yes, "total": total,
             "ratio": c_yes / total if total else None, "unresolved": c_unres},
            {"metric": "coarse_grained_novelty", "yes": n_yes, "total": len(novelty_rows),
             "ratio": n_yes / len(novelty_rows) if novelty_rows else None,
             "unresolved": n_unres},
            # Not a score. Recorded so a run of uncleaned generations cannot be
            # mistaken for one the paper would recognise.
            {"metric": "rows_not_cleaned", "yes": uncleaned, "total": total,
             "ratio": uncleaned / total if total else None, "unresolved": 0},
        ]
        return pd.DataFrame(rows)
