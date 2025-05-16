import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
from src.utils import load_json, save_json, setup_logger
from src.inference.inference_driver import InferenceDriver

class CreativeIndexInference(InferenceDriver):
    """Handles inference for the creative index task (book, speech, poem) using vLLM."""

    def __init__(self, config):
        super().__init__(config)

    def create_batched_prompt(self, data):
        """Prepares prompts for batch inference, keeping metadata."""
        all_prompt_data = []

        for sample in tqdm(data, desc="Preparing prompts"):
            problem_id = sample["meta_data"]["id"]
            dataset_name = sample["meta_data"]["dataset"]
            prompt = sample["input"]["text"]

            all_prompt_data.append({
                "prompt": prompt,
                "problem_id": problem_id,
                "dataset": dataset_name
            })

        return all_prompt_data

    def parse_vllm_outputs(self, vllm_results):
        """Processes vLLM outputs and returns structured results."""
        parsed_results = []
        for data in vllm_results:
            parsed_results.append({
                "prompt": data["prompt"],
                "prompt_id": data["problem_id"],
                "dataset": data["dataset"],
                "response": data["raw_output"].outputs[0].text,
                "model_name": self.config["model_name"]
            })
        return parsed_results

    def vllm_batch_inference(self, all_prompt_data):
        print(self.sampling_params)
        vllm_outputs = self.llm.generate(
            [d['prompt'] for d in all_prompt_data],
            self.sampling_params
        )

        for i in range(len(all_prompt_data)):
            all_prompt_data[i]['raw_output'] = vllm_outputs[i]
        return all_prompt_data
    
    def api_inference(self, prompt_data):
        # becareful
        prompt = prompt_data["prompt"]
        response = self.api_model.generate_response(prompt)
        return {
            "prompt": prompt,
            "prompt_id": prompt_data["problem_id"],
            "dataset": prompt_data["dataset"],
            "response": response,
            "model_name": self.config["model_name"]
        }

    def inference(self):
        """Loads dataset, prepares prompts, runs batched inference, and saves results."""
        print(self.config['task'])
        data_path = self.config["file_paths"]["dataset"]
        data = load_json(data_path)

        # Use full dataset or a portion
        portion = self.config.get("portion", 1)
        amount_used = int(len(data) * portion)
        # need to revise back later
        data = data[:amount_used][:100]
        
        if self.model_name in self.open_source_models:
            all_prompt_data = self.create_batched_prompt(data)
            vllm_results = self.vllm_batch_inference(all_prompt_data)
            parsed_results = self.parse_vllm_outputs(vllm_results)

        elif self.model_name in self.closed_source_model:
            all_prompt_data = self.create_batched_prompt(data)
            
            api_results = []
            for prompt_data in tqdm(all_prompt_data, desc="Running API inference"):
                result = self.api_inference(prompt_data)
                api_results.append(result)

            parsed_results = api_results
            
        else:
            raise NotImplementedError
            # Save results
        output_dir = self.config["file_paths"]["generation"]
        output_file = os.path.join(output_dir, f"{self.config['task']}.json")
        os.makedirs(output_dir, exist_ok=True)
        save_json(parsed_results, output_file)

        print(f"Results saved to {os.path.abspath(output_file)}")
        return parsed_results
