"""TTCT evaluation, ported onto the Driver/EvalDriver contract.

Reimplements ``ttct/src/evaluation/ttct_evaluation.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo). Preserved: the
rubric-driven judge prompt, the four traits, the ``### Scores ###`` block, and
the rule that only ``cot`` is judged. Changed, and why:

* **``tensor_parallel_size`` is no longer 4.** The original pinned it, so an
  open-weight judge could not run on any allocation that was not exactly four
  GPUs. ``Driver`` derives it once, from what is actually visible.
* **``basefile.csv`` and the json->csv->json round trip are gone.** They existed
  only to read ``question_type``, which ``meta_data`` already carries. Verified
  before removal: the csv and ``ttct.json`` agree on ``question_type`` for all
  700 rows and on prompt text in 2099 of 2100 fields (the exception is one
  leading space).
* **``PAIRWISE_RESPONSES`` is imported.** The original referenced it without
  importing it, so ``-pairwise True`` raised ``NameError`` on the first row.
* **The evaluation output keeps ``meta_data``.** The original dropped it, which
  left the loader joining evaluation to inference on ``(question_type, prompt
  text)`` for want of an id. ``question_type`` and ``input`` are still written,
  so that join keeps working.

Score extraction matches ``result_analysis/loader.py`` exactly, which in turn
matches the notebook that produced the published human-agreement numbers.
"""
import re

import pandas as pd

from src.evaluation.eval_driver import EvalDriver
from src.utils.helpers import load_json, llm_batch_inference
from src.prompt_engineering.ttct_rubrics import ttct_rubrics
from src.prompt_engineering.ttct_demo import llmj_demo_data
from src.prompt_engineering.ttct_eval_prompts import (
    EVAL_PROMPT,
    EVAL_W_SUMMARY_PROMPT,
    EXAMPLE_PROMPT,
    PAIRWISE_PROMPT,
    PAIRWISE_RESPONSES,
)

SKIPPED = "SKIPPED"
TRAITS = ("Fluency", "Flexibility", "Originality", "Elaboration")
SCALE_KEYS = ("1", "2", "3", "4", "5")
_SCORE_RE = re.compile(r"(\w+): (\d+)")

# Originality is prompted for but excluded from the published aggregate: human
# annotator agreement on it was too low to trust a judge with (appendix,
# LLM-as-a-Judge Design Details). It is scored and stored, never summed.
REPORTED_TRAITS = ("Fluency", "Flexibility", "Elaboration")


def _rubric_fields(question_type, demo_examples=""):
    """The 10 rubric placeholders shared by every TTCT judge template."""
    rubric = ttct_rubrics[question_type]
    fields = {"name": rubric["name"], "definition": rubric["definition"]}
    for trait in TRAITS:
        fields["{}_rubric".format(trait.lower())] = rubric["dimensions"][trait]
        fields["{}_details".format(trait.lower())] = "".join(
            "{}: {}\n".format(key, rubric["scale"][key][trait]) for key in SCALE_KEYS
        )
    # Always supplied: EVAL_PROMPT carries {demo_examples} whether or not demos
    # are on, and PAIRWISE_PROMPT has no such placeholder -- str.format ignores
    # the extra key, but a missing one is a KeyError.
    fields["demo_examples"] = demo_examples
    return fields


def parse_ttct_scores(eval_output):
    """Pull the four trait scores out of one judge response.

    Identical to ``result_analysis.loader._parse_ttct_scores``: read after the
    *first* ``### Scores ###`` marker, fall back to the whole response when the
    judge omits it, and keep only the four known traits.
    """
    if not isinstance(eval_output, str):
        return {}
    body = (
        eval_output.split("### Scores ###")[1]
        if "### Scores ###" in eval_output
        else eval_output
    )
    found = {name: int(value) for name, value in _SCORE_RE.findall(body)}
    return {trait.lower(): found[trait] for trait in TRAITS if trait in found}


class TTCTEval(EvalDriver):

    def __init__(self, config={}):
        super().__init__(config)

    def _demo_block(self, question_type):
        if not self.config.get("demo", False) or question_type not in llmj_demo_data:
            return ""
        demos = llmj_demo_data[question_type]
        block = (
            "\n====================\nEXAMPLE RATINGS\n===================="
            "\nHere are some example ratings for your reference:"
        )
        for level in ("highest", "median", "lowest"):
            block += "Score level: {}:\n".format(level)
            block += EXAMPLE_PROMPT.format(**demos[level]) + "\n\n"
        return block

    def _pairwise_prompt(self, item, question_type):
        """Judge the response against a reference answer, in a random order.

        ``target`` records which slot the model's own answer went into, so the
        parse can tell a win from a loss.
        """
        import random

        cot_question = item["input"]["text_cot"]
        cot_prediction = item["output"]["text_cot"]
        if question_type in llmj_demo_data:
            ref = llmj_demo_data[question_type]["highest"]
            ref_question, ref_prediction = ref["question"], ref["answer"]
        else:
            ref_question, ref_prediction = "", ""

        if random.choice([0, 1]):
            target = "A"
            pairwise_data = PAIRWISE_RESPONSES.format(
                question_a=cot_question, response_a=cot_prediction,
                question_b=ref_question, response_b=ref_prediction,
            )
        else:
            target = "B"
            pairwise_data = PAIRWISE_RESPONSES.format(
                question_a=ref_question, response_a=ref_prediction,
                question_b=cot_question, response_b=cot_prediction,
            )
        head = PAIRWISE_PROMPT.format(**_rubric_fields(question_type))
        return head + pairwise_data, target

    def create_batched_prompt(self, input_data, prompt_format="cot"):
        """One prompt_data per scored row.

        Rows the inference marked ``skip``, or whose requested variant came back
        ``SKIPPED``, produce no prompt at all -- the original built the string
        anyway and then compared it to the sentinel.
        """
        pairwise = self.config.get("pairwise", False)
        template = EVAL_W_SUMMARY_PROMPT if self.config.get("summary", False) else EVAL_PROMPT

        all_prompt_data = []
        for index, item in enumerate(input_data):
            if item.get("skip", False):
                continue
            question_type = item["meta_data"]["question_type"]
            if question_type not in ttct_rubrics:
                continue
            response = (item.get("output") or {}).get("text_{}".format(prompt_format))
            if not isinstance(response, str) or response.strip().upper() == SKIPPED:
                continue

            target = None
            if pairwise:
                prompt_text, target = self._pairwise_prompt(item, question_type)
            else:
                head = template.format(
                    **_rubric_fields(question_type, self._demo_block(question_type))
                )
                prompt_text = (
                    "{}\nQuestion: {}\nAnswer: {}\n\n"
                    "ANSWER RATINGS ON FOUR DIMENSIONS:".format(
                        head, item["input"]["text_{}".format(prompt_format)], response
                    )
                )

            all_prompt_data.append({
                "prompt_id": (item["meta_data"]["id"], index, prompt_format),
                "prompt_text": prompt_text,
                "messages": [{"role": "user", "content": prompt_text}],
                "data": item,
                "row_index": index,
                "pairwise_target": target,
            })
        return all_prompt_data

    def parse_llm_outputs(self, llm_results):
        for result in llm_results:
            result["cleaned_output"] = result["raw_output"]
            result["scores"] = parse_ttct_scores(result["raw_output"])
        return llm_results

    def evaluation(self):
        prompt_format = self.config.get("eval_prompt_format", "cot")
        input_data_dir = self.config["eval_input_dir"]
        input_data = load_json(input_data_dir)
        if not input_data:
            raise FileNotFoundError("TTCT inference output not found: {}".format(input_data_dir))

        all_prompt_data = self.create_batched_prompt(input_data, prompt_format)
        print("=> judging {} row(s) on '{}'".format(len(all_prompt_data), prompt_format))

        llm_results = []
        if all_prompt_data:
            llm_results = llm_batch_inference(
                self.llm,
                all_prompt_data,
                self.use_open_model,
                self.sampling_params,
                self.config,
            )
            llm_results = self.parse_llm_outputs(llm_results)

        verdict_by_row = {r["row_index"]: r for r in llm_results}
        eval_output_cleaned = []
        for index, item in enumerate(input_data):
            result = verdict_by_row.get(index)
            if result is None:
                continue
            eval_output_cleaned.append({
                "meta_data": item["meta_data"],
                # Kept alongside meta_data because the loader joins on these two.
                "question_type": item["meta_data"]["question_type"],
                "input": item["input"],
                "output": item["output"],
                "evaluation": {
                    "text_basic": SKIPPED,
                    "text_instructive": SKIPPED,
                    "text_cot": SKIPPED,
                    "text_{}".format(prompt_format): result["cleaned_output"],
                },
                "scores": result["scores"],
                "pairwise_target": result["pairwise_target"],
            })

        return self.generate_eval_report(eval_output_cleaned), eval_output_cleaned

    def generate_eval_report(self, eval_outputs):
        """One row per (question_type, trait) with the mean score and its n.

        Originality is scored and kept but excluded from ``reported``: the paper
        drops it from aggregation because human agreement on it was too low.
        """
        rows = []
        for item in eval_outputs:
            for trait, score in (item.get("scores") or {}).items():
                rows.append({
                    "id": item["meta_data"]["id"],
                    "question_type": item["question_type"],
                    "trait": trait,
                    "score": score,
                })
        if not rows:
            return pd.DataFrame(columns=["question_type", "trait", "mean_score", "n", "reported"])

        df = pd.DataFrame(rows)
        report = (
            df.groupby(["question_type", "trait"])["score"]
            .agg(mean_score="mean", n="count")
            .reset_index()
        )
        report["reported"] = report["trait"].isin(
            [t.lower() for t in REPORTED_TRAITS]
        )
        return report
