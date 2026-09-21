"""Creativity Index inference, ported onto the Driver/InferenceDriver contract.

Reimplements ``math_n_index/src/inference/creative_index_inference.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo),
which was already a near-copy of this contract -- same three abstract methods,
but with its own model loading, its own batch helper, and a parser that reached
into vLLM's result objects. All three now come from the base class.

The task is completion, not instruction-following: the model is handed a first
line and asked to continue it. There are three domains (book, poem, speech) and
the paper scores the first 100 examples of each; ``book`` has exactly 100 rows,
``poem`` 247 and ``speech`` 235, so the cap does real work on two of the three.

Hyperparameters are the paper's (appendix, Creativity Index / Experiment
Configurations): temperature 1, max_tokens 288, top_p 0.9. They are unusual
enough -- 288 is a very short cap, and 0.9 the only non-1 top_p in the suite --
that the config states them rather than leaning on any default.
"""
from src.inference.inference_driver import InferenceDriver
from src.utils.helpers import data_path, load_json, llm_batch_inference

DOMAINS = ("book", "poem", "speech")

# The paper: "We use the first 100 examples in generation and evaluation."
DEFAULT_SUBSET_SIZE = 100


class CreativityIndexInference(InferenceDriver):

    def __init__(self, config={}):
        super().__init__(config)

    def create_batched_prompt(self, input_data, domain):
        all_prompt_data = []
        for dp in input_data:
            prompt_text = dp["input"]["text"]
            all_prompt_data.append({
                "prompt_id": dp["meta_data"]["id"],
                "prompt_text": prompt_text,
                "messages": [{"role": "user", "content": prompt_text}],
                "data": dp,
                "domain": domain,
            })
        return all_prompt_data

    def parse_llm_outputs(self, llm_results):
        """Closed-source models preface completions; open-source ones do not.

        The paper says the prefaces ("Certainly", "Here is...") were stripped by
        hand. Nothing is stripped here -- a hand edit cannot be reproduced, and
        guessing at it would change the scored text. The raw generation is what
        gets scored, and the difference is recorded rather than papered over.
        """
        for result in llm_results:
            result["cleaned_output"] = result["raw_output"]
        return llm_results

    def inference(self):
        data_dir = self.config.get(
            "input_data_dir", data_path("creativity_index")
        )
        domains = self.config.get("domains", list(DOMAINS))
        subset_size = self.config.get("subset_size", DEFAULT_SUBSET_SIZE)
        test_size = self.config.get("test_size", -1)

        results = []
        for domain in domains:
            path = "{}/{}_prompt.json".format(data_dir, domain)
            input_data = load_json(path)
            if not input_data:
                raise FileNotFoundError("creativity_index input not found: {}".format(path))

            input_data = input_data[:subset_size]
            if test_size and test_size > 0:
                input_data = input_data[:int(test_size)]

            all_prompt_data = self.create_batched_prompt(input_data, domain)
            print("=> {}: {} prompt(s)".format(domain, len(all_prompt_data)))

            llm_results = llm_batch_inference(
                self.llm,
                all_prompt_data,
                self.use_open_model,
                self.sampling_params,
                self.config,
            )
            llm_results = self.parse_llm_outputs(llm_results)

            for result in llm_results:
                results.append({
                    "meta_data": result["data"]["meta_data"],
                    "domain": domain,
                    # `prompt` and `response` are the names the evaluation and the
                    # loader already read; keeping them avoids a rename that would
                    # buy nothing and break both.
                    "prompt": result["prompt_text"],
                    "prompt_id": result["prompt_id"],
                    "response": result["cleaned_output"],
                    "model_name": self.model_name,
                })

        return results
