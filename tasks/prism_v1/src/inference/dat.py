"""DAT inference: one fixed prompt, sampled many times.

Reimplements ``neocoder_dat/steps/inference_dat.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo), which was a click
script over a CodeGenerator rather than a Driver subclass.

DAT has no dataset. The paper generates 100 samples per model from a single
prompt and scores the words in each. Sampling has to be *on* for that to mean
anything -- a hundred greedy samples would be a hundred identical answers -- so
the config carries the paper's temperature 0.75, and the config, not a CLI
default, is where it lives. The original script defaulted to temperature 0.7 and
max_tokens 512, both of which disagree with the appendix (0.75 and 4096).

**The prompt has a typo.** The string that has always been sent reads
"as irrelevantfrom each other", with no space; the appendix prints it with one.
The typo is kept as the default because it is what produced the published
numbers, and ``prompt`` in the config overrides it. Fixing it silently would
change every DAT score without leaving a trace.
"""
from src.inference.inference_driver import InferenceDriver
from src.utils.helpers import llm_batch_inference

# Verbatim from neocoder_dat/src/utils/configs.py, typo included.
DAT_PROMPT = (
    "Please write 10 nouns in English that are as irrelevantfrom each other as "
    "possible, in all meanings and uses of the words. Please note that the words "
    "you write should have only single word, only nouns (e.g., things, objects, "
    "concepts), and no proper nouns (e.g., no specific people or places). Your answer:"
)

DEFAULT_REPEAT = 100


class DATInference(InferenceDriver):

    def __init__(self, config={}):
        super().__init__(config)

    def create_batched_prompt(self, prompt_text, repeat):
        return [{
            "prompt_id": "dat_{}".format(i),
            "prompt_text": prompt_text,
            "messages": [{"role": "user", "content": prompt_text}],
            "data": {"meta_data": {"dataset": "dat", "id": "dat_{}".format(i)}},
            "sample_index": i,
        } for i in range(repeat)]

    def parse_llm_outputs(self, llm_results):
        """Word extraction belongs to the evaluation, which owns the heuristics."""
        for result in llm_results:
            result["cleaned_output"] = result["raw_output"]
        return llm_results

    def inference(self):
        prompt_text = self.config.get("prompt", DAT_PROMPT)
        repeat = int(self.config.get("repeat", DEFAULT_REPEAT))
        test_size = self.config.get("test_size", -1)
        if test_size and test_size > 0:
            repeat = min(repeat, int(test_size))

        if float(self.config.get("temperature", 0.75)) == 0:
            raise ValueError(
                "DAT samples one prompt {} times; at temperature 0 every sample is "
                "identical and the score is meaningless.".format(repeat)
            )

        all_prompt_data = self.create_batched_prompt(prompt_text, repeat)
        print("=> DAT: {} sample(s) of one prompt".format(repeat))

        llm_results = llm_batch_inference(
            self.llm,
            all_prompt_data,
            self.use_open_model,
            self.sampling_params,
            self.config,
        )
        llm_results = self.parse_llm_outputs(llm_results)

        return [{
            "meta_data": r["data"]["meta_data"],
            "prompt_id": r["prompt_id"],
            "sample_index": r["sample_index"],
            "prompt": prompt_text,
            # `output` is the key the DAT word parser reads.
            "output": r["cleaned_output"],
            "model_name": self.model_name,
        } for r in llm_results]
