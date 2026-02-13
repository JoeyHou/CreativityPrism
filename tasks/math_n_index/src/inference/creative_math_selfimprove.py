import logging
import os
from tqdm import tqdm
from src.prompt_engineering.creative_math_prompts import (
    load_novel_solution_generation_prompt,
    load_feedback_prompt,
    load_refinement_with_feedback_prompt
)
from src.utils.helpers import load_json, save_json
from src.inference.inference_driver import InferenceDriver

class CreativeMathImprove(InferenceDriver):
    """Handles inference for the creative math task using vLLM, with self-improvement."""

    def __init__(self, config):
        super().__init__(config)

    def create_batched_prompt(self, data):
        """Prepares prompts and associated metadata for batch inference."""
        all_prompt_data = []

        for question_number, sample in tqdm(enumerate(data), desc="Preparing prompts"):
            problem_id = sample["meta_data"]["id"]
            dataset_name = sample["meta_data"]["dataset"]
            problem = sample["input"]["text"]
            solutions = sample["input"]["others"]["references"]["solutions"]
            n = len(solutions)

            for k in range(1, n + 1):
                prompt = load_novel_solution_generation_prompt(problem, solutions, k)
                all_prompt_data.append({
                    "prompt": prompt,
                    "problem_id": problem_id,
                    "problem": problem,
                    "question_number": question_number,
                    "dataset": dataset_name,
                    "k": k,
                    "n": n,
                    "ground_truth_solutions": solutions[:k],
                    "all_solutions": solutions,
                    "full_solution_set": solutions
                })

        return all_prompt_data

    def parse_vllm_outputs(self, vllm_results):
        """Extracts responses from vLLM outputs, including problem text."""
        parsed_results = []
        for data in vllm_results:
            parsed_results.append({
                "problem": data["problem"],
                "problem_id": data["problem_id"],
                "question_number": data["question_number"],
                "dataset": data["dataset"],
                "k": data["k"],
                "n": data["n"],
                "ground_truth_solutions": data["ground_truth_solutions"],
                "response": data["raw_output"].outputs[0].text,
                "all_solutions": data["all_solutions"],
                "full_solution_set": data.get("full_solution_set", data["all_solutions"])
            })
        return parsed_results

    def vllm_batch_inference(self, all_prompt_data):
        """Runs batch inference using vLLM."""
        vllm_outputs = self.llm.generate(
            [d['prompt'] for d in all_prompt_data],
            self.sampling_params
        )

        for i in range(len(all_prompt_data)):
            all_prompt_data[i]['raw_output'] = vllm_outputs[i]
        return all_prompt_data

    def api_batch_inference(self, all_prompt_data):
        """Runs batch inference using API."""
        for data in tqdm(all_prompt_data, desc="Running API inference"):
            response = self.api_model.generate_response(data['prompt'])
            data['raw_output'] = {"outputs": [{"text": response}]}
        return all_prompt_data

    def inference(self):
        """Runs the full iterative self-improvement inference process."""
        data_path = self.config["file_paths"]["dataset"]
        data = load_json(data_path)

        portion = self.config.get("portion", 1.0)
        amount_used = int(len(data) * portion)
        data = data[:amount_used]

        print("\n================= generating initial prompts =================")
        all_prompt_data = self.create_batched_prompt(data)

        print("\n================= doing initial inference =================")
        if self.model_name in self.open_source_models:
            vllm_results = self.vllm_batch_inference(all_prompt_data)
        elif self.model_name in self.closed_source_model:
            vllm_results = self.api_batch_inference(all_prompt_data)
        else:
            raise NotImplementedError("Model type not supported")

        parsed_results = self.parse_vllm_outputs(vllm_results)

        for result in parsed_results:
            result["iteration_0_response"] = result["response"]

        n_iter = self.config.get("n_iterations", 3)

        for iteration in range(1, n_iter):
            print(f"\n================= refinement iteration {iteration} =================")

            feedback_prompt_data = []
            for result in parsed_results:
                problem = result["problem"]
                references = result["ground_truth_solutions"]
                prev_response = result["response"]
                prompt = load_feedback_prompt(problem, references, prev_response)
                feedback_prompt_data.append({
                    "prompt": prompt,
                    "problem": problem,
                    "problem_id": result["problem_id"],
                    "question_number": result["question_number"],
                    "dataset": result["dataset"],
                    "k": result["k"],
                    "n": result["n"],
                    "ground_truth_solutions": references,
                    "all_solutions": result["all_solutions"],
                    "full_solution_set": result["full_solution_set"]
                })

            if self.model_name in self.open_source_models:
                feedback_results = self.vllm_batch_inference(feedback_prompt_data)
            else:
                feedback_results = self.api_batch_inference(feedback_prompt_data)
            parsed_feedback = self.parse_vllm_outputs(feedback_results)

            for i in range(len(parsed_results)):
                parsed_results[i][f"iteration_{iteration}_feedback"] = parsed_feedback[i]["response"]

            refinement_prompt_data = []
            for i, result in enumerate(parsed_results):
                problem = result["problem"]
                references = result["ground_truth_solutions"]

                history = f"Attempt 0:\n{result['iteration_0_response']}\n"
                for t in range(1, iteration + 1):
                    fb = result.get(f"iteration_{t}_feedback", "")
                    resp = result.get(f"iteration_{t}_response", "")
                    history += f"\nFeedback {t}:\n{fb}\nAttempt {t}:\n{resp}\n"

                prompt = load_refinement_with_feedback_prompt(problem, references, history)

                refinement_prompt_data.append({
                    "prompt": prompt,
                    "problem": problem,
                    "problem_id": result["problem_id"],
                    "question_number": result["question_number"],
                    "dataset": result["dataset"],
                    "k": result["k"],
                    "n": result["n"],
                    "ground_truth_solutions": references,
                    "all_solutions": result["all_solutions"],
                    "full_solution_set": result["full_solution_set"]
                })

            if self.model_name in self.open_source_models:
                refined_results = self.vllm_batch_inference(refinement_prompt_data)
            else:
                refined_results = self.api_batch_inference(refinement_prompt_data)
            parsed_refinements = self.parse_vllm_outputs(refined_results)

            for i in range(len(parsed_results)):
                parsed_results[i]["response"] = parsed_refinements[i]["response"]
                parsed_results[i][f"iteration_{iteration}_response"] = parsed_refinements[i]["response"]

        print("\n================= saving results =================")
        output_dir = self.config["file_paths"]["generation"]
        output_file = f"{output_dir}/{self.config['model_name']}_self_refine_{n_iter}.json"
        os.makedirs(output_dir, exist_ok=True)
        save_json(parsed_results, output_file)

        logging.info(f"Final results saved to {output_file}")
        return parsed_results