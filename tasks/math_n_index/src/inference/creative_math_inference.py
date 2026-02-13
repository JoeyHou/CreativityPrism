import logging
import os
from tqdm import tqdm
from src.prompt_engineering.creative_math_prompts import load_novel_solution_generation_prompt
from src.utils.helpers import load_json, save_json
from src.inference.inference_driver import InferenceDriver

class CreativeMathInference(InferenceDriver):
    """Handles inference for the creative math task using vLLM or API."""

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
                    "all_solutions": solutions
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
                "all_solutions": data["all_solutions"]
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

    def api_inference(self, prompt_data):
        """Runs single-sample inference via API."""
        prompt = prompt_data["prompt"]
        response = self.api_model.generate_response(prompt)
        return {
            "problem": prompt_data["problem"],
            "problem_id": prompt_data["problem_id"],
            "question_number": prompt_data["question_number"],
            "dataset": prompt_data["dataset"],
            "k": prompt_data["k"],
            "n": prompt_data["n"],
            "ground_truth_solutions": prompt_data["ground_truth_solutions"],
            "response": response,
            "all_solutions": prompt_data["all_solutions"]
        }

    def inference(self):
        """Loads dataset, prepares prompts, and runs either vLLM or API inference."""
        data_path = self.config["file_paths"]["dataset"]
        data = load_json(data_path)

        portion = self.config.get("portion", 1.0)
        amount_used = int(len(data) * portion)
        data = data[:amount_used]

        print("\n================= generating prompts =================")
        all_prompt_data = self.create_batched_prompt(data)

        if self.model_name in self.open_source_models:
            print("\n================= doing vLLM inference =================")
            vllm_results = self.vllm_batch_inference(all_prompt_data)
            parsed_results = self.parse_vllm_outputs(vllm_results)

        elif self.model_name in self.closed_source_model:
            print("\n================= doing API inference =================")
            parsed_results = []
            for prompt_data in tqdm(all_prompt_data, desc="Running API inference"):
                result = self.api_inference(prompt_data)
                parsed_results.append(result)

        else:
            raise NotImplementedError("Model type not supported")

        print("\n================= saving results =================")
        output_dir = self.config["file_paths"]["generation"]
        output_file = f"{output_dir}/{self.config['model_name']}.json"
        os.makedirs(output_dir, exist_ok=True)
        save_json(parsed_results, output_file)

        logging.info(f"Results saved to {output_file}")
        return parsed_results
