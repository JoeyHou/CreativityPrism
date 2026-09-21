"""NeoCoder inference: solve one Codeforces problem under growing denial constraints.

Reimplements ``neocoder_dat/steps/inference_dp.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo) plus the two generator
classes behind it. The originals were click scripts over a torch ``DataLoader``
and a ``CodeGenerator`` hierarchy that branched on provider type; the dataset is
199 problems of plain JSON, so none of that is needed to hand prompts to a model.

**Denial prompting here is not adaptive.** Each problem ships six ready-made
``problem_statements``: round 0 is the bare problem, and rounds 1-5 prepend a
growing "DO NOT use the following techniques" list taken from
``constraints_list``. The constraints are baked into the dataset, not derived
from what this model just wrote, so every model is denied the same techniques in
the same order.

The constraint sets repeat: for 1901A the six rounds are
``[og], [for loop], [for loop], [for loop, while loop], [for loop, while loop],
[for loop, while loop, sorting]`` -- four distinct sets across six rounds. The
scoring step drops rounds whose constraints match the previous one, which is
worth knowing before reading anything called "@5".

Hyperparameters are the paper's (appendix, NeoCoder / Experiment Configurations):
temperature 0.75, max_tokens 4096, top_p 1. The original CLI defaulted to 0.7 and
512.
"""
from src.inference.inference_driver import InferenceDriver
from src.utils.helpers import data_path, load_json, llm_batch_inference

DEFAULT_DP_ROUNDS = 5


class NeoCoderInference(InferenceDriver):

    def __init__(self, config={}):
        super().__init__(config)

    def create_batched_prompt(self, input_data, dp_rounds):
        """Flatten (problem, round) into one list so a single batch covers them all.

        The original grouped by problem and generated per problem; flattening
        gives the engine every prompt at once, which is where vLLM's throughput
        comes from.
        """
        all_prompt_data = []
        for problem in input_data:
            statements = problem["problem_statements"][:dp_rounds + 1]
            for dp_idx, statement in enumerate(statements):
                all_prompt_data.append({
                    "prompt_id": "{}|dp{}".format(problem["problem_id"], dp_idx),
                    "prompt_text": statement,
                    "messages": [{"role": "user", "content": statement}],
                    "data": problem,
                    "problem_id": problem["problem_id"],
                    "dp_idx": dp_idx,
                })
        return all_prompt_data

    def parse_llm_outputs(self, llm_results):
        """Code extraction belongs to the evaluation.

        Both the generator and the evaluator had their own extractor, and the
        evaluator's is the one that decides what gets executed. Leaving the raw
        text here means there is exactly one place where a change to parsing can
        change a score.
        """
        for result in llm_results:
            result["cleaned_output"] = result["raw_output"]
        return llm_results

    def inference(self):
        input_data_dir = self.config.get(
            "input_data_dir", data_path("neocoder/NeoCoder.json")
        )
        input_data = load_json(input_data_dir)
        if not input_data:
            raise FileNotFoundError("NeoCoder input not found: {}".format(input_data_dir))

        test_size = self.config.get("test_size", -1)
        if test_size and test_size > 0:
            input_data = input_data[:int(test_size)]

        dp_rounds = int(self.config.get("dp_rounds", DEFAULT_DP_ROUNDS))
        all_prompt_data = self.create_batched_prompt(input_data, dp_rounds)
        print("=> {} problem(s) x {} round(s) -> {} generation(s)".format(
            len(input_data), dp_rounds + 1, len(all_prompt_data)))

        llm_results = llm_batch_inference(
            self.llm,
            all_prompt_data,
            self.use_open_model,
            self.sampling_params,
            self.config,
        )
        llm_results = self.parse_llm_outputs(llm_results)

        # Regroup into one record per problem: the evaluation walks rounds in
        # order and needs constraints_list beside them.
        by_problem = {}
        for result in llm_results:
            record = by_problem.setdefault(result["problem_id"], {
                "meta_data": {"dataset": "neocoder", "id": result["problem_id"]},
                "problem_id": result["problem_id"],
                "problem_statements": [],
                "outputs": [],
                "constraints": result["data"].get("constraints_list", []),
                "model_name": self.model_name,
            })
            record["problem_statements"].append(result["prompt_text"])
            record["outputs"].append(result["cleaned_output"])

        return [by_problem[p["problem_id"]] for p in input_data if p["problem_id"] in by_problem]
