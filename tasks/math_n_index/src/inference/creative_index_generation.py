import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from src.models import ModelWrapperVLLM
from src.utils import load_json, setup_logger
from api_warpper import ModelWrapper

def load_config():
    file_path = "configs/creative_index_config_old.json"
    with open(file_path, "r") as file:
        return json.load(file)

def main():
    # TODO: revise these
    parser = argparse.ArgumentParser(
        description="Run the novel solution generation program."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Llama-3.1-8B",
        help="The model used to generate novel solutions.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="book",
        help="The task to run, including book, speech, and poem."
    )
    parser.add_argument(
        "--portion",
        type=int,
        default=1,
        help="The amount of data to use."
    )
    args = parser.parse_args()
    model_name = args.model_name
    model_task = args.task

    config = load_config()
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # TODO: do I need to follow this format? this file is used in evaluation.
    log_file = os.path.join(
        config["logging"]["log_dir"],
        f"generation_{model_name}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting the {model_task} generation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")
    # TODO: please check
    model = ModelWrapperVLLM(model_name)

    data_path = config["file_paths"]["dataset"][model_task]
    # TODO: check data loading problem
    # for testing 
    data = load_json(data_path)
    amount_used = int(len(data) * args.portion)
    data = data[:amount_used]
    
    results = []
    for sample in tqdm(data):
        problem_id = sample["meta_data"]["id"]
        dataset_name = sample["meta_data"]["dataset"] 

        prompt = sample["input"]["text"] 
        
        response = model.generate_response(prompt)

        results.append({
                "prompt": prompt,
                "prompt_id": problem_id,
                "dataset": dataset_name,
                "response": response,
                "model_name": model_name
        })

    output_dir = config["file_paths"]["generation"][model_task]
    output_file = os.path.join(output_dir, f"{model_name}_{model_task}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    
    logger.info(f"Results saved to {os.path.abspath(output_file)}")



if __name__ == "__main__":
    main()
