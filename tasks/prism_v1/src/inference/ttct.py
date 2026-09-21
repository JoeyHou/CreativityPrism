"""TTCT inference, ported onto the Driver/InferenceDriver contract.

Reimplements ``ttct/src/inference/ttct_inference.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo). Behaviour that the
original had is preserved; what changed, and why:

* **The model is loaded by ``Driver``, not here.** The original built its own
  ``LLM(...)`` and derived ``tensor_parallel_size`` from ``torch.cuda.device_count()``.
  Its evaluation twin hardcoded 4, which cannot run on a one-GPU allocation.
  Both now inherit the base class's single loader.
* **Hyperparameters come from the config**, and the config carries the paper's
  numbers (appendix, TTCT / Experiment Configurations): temperature 0.75,
  max_tokens 4096, top_p 1. The original module pinned ``MAX_TOKENS = 3072`` and
  defaulted ``-temp 1``, which disagrees with the paper on both counts.
* **``basefile.csv`` is gone.** The original round-tripped JSON to CSV and back
  purely to read ``question_type`` out of the csv -- but ``question_type`` is
  already in ``meta_data``, and the csv's prompt text matches ``ttct.json`` in
  2099 of 2100 fields (the one difference is a leading space). See TTCTEval.

The three prompt formats (basic / instructive / cot) live side by side in one
row of the dataset, so a "prompt" here is a (row, format) pair. Only ``cot`` is
judged, so the other two default to off and are recorded as ``SKIPPED``.
"""
from collections import Counter

from src.inference.inference_driver import InferenceDriver
from src.utils.helpers import data_path, load_json, llm_batch_inference

PROMPT_FORMATS = ("basic", "instructive", "cot")

# 5 of the 7 question types. Story Writing and Just-suppose were dropped because
# annotators found them too subjective to judge (appendix, LLM-as-a-Judge Design
# Details), so their 200 rows ride through unscored as SKIPPED.
DEFAULT_SUBSET = [
    "1_unusual_uses", "5_common_problems", "4_situation", "6_improvement", "2_consequences",
]

SKIPPED = "SKIPPED"


class TTCTInference(InferenceDriver):

    def __init__(self, config={}):
        super().__init__(config)

    def create_batched_prompt(self, input_data, prompt_format):
        """One prompt_data per row that is in-subset and under the sample cap.

        ``num_samples`` caps *per question type*, not overall -- the original
        counted that way and the runner's ``--limit N`` inherits the meaning, so
        ``--limit 2`` is 10 prompts across 5 types rather than 2.
        """
        subset = self.config.get("subset", DEFAULT_SUBSET)
        num_samples = self.config.get("num_samples", -1)

        all_prompt_data = []
        kept = Counter()
        for index, dp in enumerate(input_data):
            question_type = dp["meta_data"]["question_type"]
            if subset and question_type not in subset:
                continue
            kept[question_type] += 1
            if num_samples > 0 and kept[question_type] > num_samples:
                continue

            prompt_text = dp["input"]["text_{}".format(prompt_format)]
            all_prompt_data.append({
                "prompt_id": (dp["meta_data"]["id"], index, prompt_format),
                "prompt_text": prompt_text,
                "messages": [{"role": "user", "content": prompt_text}],
                "data": dp,
                "row_index": index,
            })
        return all_prompt_data

    def parse_llm_outputs(self, llm_results):
        """TTCT scores the raw generation; there is nothing to strip."""
        for result in llm_results:
            result["cleaned_output"] = result["raw_output"]
        return llm_results

    def inference(self):
        input_data_dir = self.config.get(
            "input_data_dir", data_path("ttct/ttct.json")
        )
        input_data = load_json(input_data_dir)
        if not input_data:
            raise FileNotFoundError("TTCT input not found: {}".format(input_data_dir))

        prompt_formats = self.config.get("prompt_formats", ["cot"])
        subset = self.config.get("subset", DEFAULT_SUBSET)
        self.logger.info("TTCT: {} rows, subset {}, formats {}".format(
            len(input_data), subset, prompt_formats))

        # Every row keeps an entry for every format so the output schema is fixed
        # regardless of which formats were requested or how the cap fell.
        outputs = [{fmt: SKIPPED for fmt in PROMPT_FORMATS} for _ in input_data]

        for prompt_format in PROMPT_FORMATS:
            if prompt_format not in prompt_formats:
                continue
            all_prompt_data = self.create_batched_prompt(input_data, prompt_format)
            print("=> {}: {} prompt(s)".format(prompt_format, len(all_prompt_data)))
            if not all_prompt_data:
                continue

            llm_results = llm_batch_inference(
                self.llm,
                all_prompt_data,
                self.use_open_model,
                self.sampling_params,
                self.config,
            )
            llm_results = self.parse_llm_outputs(llm_results)

            for result in llm_results:
                outputs[result["row_index"]][prompt_format] = result["cleaned_output"]

        for i, dp in enumerate(input_data):
            dp["output"] = {
                "text_{}".format(fmt): outputs[i][fmt] for fmt in PROMPT_FORMATS
            }
            # The judge skips a row unless at least one requested format produced
            # text, which is also what makes `--limit` cheap: capped rows are
            # marked, not deleted, so the file keeps one row per dataset item.
            dp["skip"] = all(outputs[i][fmt] == SKIPPED for fmt in PROMPT_FORMATS)

        return input_data
