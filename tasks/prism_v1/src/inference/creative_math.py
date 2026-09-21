"""Creative Math inference, ported onto the Driver/InferenceDriver contract.

Reimplements ``math_n_index/src/inference/creative_math_inference.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo).
Same prompt (it matches the paper appendix verbatim), same expansion, base class
does the model loading and batching.

The unit of work is not a question but a (question, k) pair: a question with n
reference solutions is asked n times, showing the first k solutions each time for
k = 1..n. Over 373 questions that is 574 generations, which is the number the
paper reports and the reason its correctness ratio is "out of 574, not 373".

Hyperparameters are the paper's (appendix, Creative Math / Experiment
Configurations): temperature 0, max_tokens 2000, top_p 1.
"""
from src.inference.inference_driver import InferenceDriver
from src.utils.helpers import data_path, load_json, llm_batch_inference
from src.prompt_engineering.creative_math_prompts import (
    load_novel_solution_generation_prompt,
)


class CreativeMathInference(InferenceDriver):

    def __init__(self, config={}):
        super().__init__(config)

    def create_batched_prompt(self, input_data):
        all_prompt_data = []
        for question_number, sample in enumerate(input_data):
            problem = sample["input"]["text"]
            solutions = sample["input"]["others"]["references"]["solutions"]
            n = len(solutions)
            for k in range(1, n + 1):
                prompt_text = load_novel_solution_generation_prompt(problem, solutions, k)
                all_prompt_data.append({
                    "prompt_id": "{}|k{}".format(sample["meta_data"]["id"], k),
                    "prompt_text": prompt_text,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "data": sample,
                    "problem": problem,
                    "question_number": question_number,
                    "k": k,
                    "n": n,
                    "ground_truth_solutions": solutions[:k],
                    "all_solutions": solutions,
                })
        return all_prompt_data

    def parse_llm_outputs(self, llm_results):
        """No cleaning here.

        The paper cleans generations with Llama-3.3-70B-Instruct before judging,
        to strip transitional phrases and the model's own claims about how novel
        its solution is. That is a separate stage with its own model and its own
        cost, not something to smuggle into a parser -- see CreativeMathEval,
        which records on every row whether cleaning actually happened.
        """
        for result in llm_results:
            result["cleaned_output"] = result["raw_output"]
        return llm_results

    def inference(self):
        input_data_dir = self.config.get(
            "input_data_dir", data_path("creative_math/creative_math.json")
        )
        input_data = load_json(input_data_dir)
        if not input_data:
            raise FileNotFoundError("creative_math input not found: {}".format(input_data_dir))

        test_size = self.config.get("test_size", -1)
        if test_size and test_size > 0:
            input_data = input_data[:int(test_size)]

        all_prompt_data = self.create_batched_prompt(input_data)
        print("=> {} question(s) -> {} (question, k) generation(s)".format(
            len(input_data), len(all_prompt_data)))

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
            "problem_id": r["data"]["meta_data"]["id"],
            "problem": r["problem"],
            "question_number": r["question_number"],
            "k": r["k"],
            "n": r["n"],
            "ground_truth_solutions": r["ground_truth_solutions"],
            "all_solutions": r["all_solutions"],
            "response": r["cleaned_output"],
            "model_name": self.model_name,
            "cleaned": False,
        } for r in llm_results]
