import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from configs import config
# TODO: we first keep the original implementation, shall we remove this?
from src.models import ModelWrapper, ModelWrapperVLLM
from src.prompt_engineering.prompts import load_novel_solution_generation_prompt
from src.utils import load_json, save_json, setup_logger

save_interval = config["experiment"][
    "save_interval"
]  # Save results after every 20 samples


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
        "--portion",
        type=int,
        default=0.04,
        help="The amount of data to use."
    )
    args = parser.parse_args()
    model_name = args.model_name

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # TODO: do I need to follow this format? this file is used in evaluation.
    log_file = os.path.join(
        config["logging"]["log_dir"],
        f"generation_{model_name}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the novel solution generation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")
    # TODO: please check
    model = ModelWrapperVLLM(model_name)

    data_path = config["file_paths"]["dataset"]
    # TODO: check data loading problem
    # for testing 
    data = load_json(data_path)
    amount_used = int(len(data) * args.portion)
    data = data[:amount_used]
    
    results = []
    for question_number, sample in tqdm(enumerate(data)): # keep the number of question as originally did
        problem_id = sample["meta_data"]["id"]
        dataset_name = sample["meta_data"]["dataset"] 

        problem = sample["input"]["text"] 
        solutions = sample["input"]["others"]["references"]["solutions"] 
        n = len(solutions) 
        
        # Iterate through different numbers of reference solutions (k = 1 to n)
        # for each question, there should be up to n generations
        for k in range(1, n + 1):
            
            prompt = load_novel_solution_generation_prompt(problem, solutions, k)  # No change needed
            response = model.generate_response(prompt)  # Check model response method

            results.append({
                "problem_id": problem_id,
                "question_number": question_number, 
                "dataset": dataset_name,
                "k": k, # need to record for calculation, num ref sol
                "n": n, # num total ref sol
                "response": response
            })

    # Save results
    output_dir = config["file_paths"]["generation"]
    output_file = os.path.join(output_dir, f"{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, output_file)
    
    logger.info(f"Results saved to {os.path.abspath(output_file)}")

# previous implementation:
'''    data = load_json(data_path)

    results = []
    for problem_id, sample in tqdm(enumerate(data)):
        problem = sample["problem"]
        solutions = list(sample["solutions"].values())  # All solutinos
        n = len(solutions)  # Total number of solutions

        # k: number of the reference solutions provided in the prompt
        # Interate k from 1, 2, until n
        for k in range(1, n + 1):
            prompt = load_novel_solution_generation_prompt(problem, solutions, k) # do not need to change
            response = model.generate_response(prompt) # TODO: please check 
            results.append(
                {"problem_id": problem_id, "k": k, "n": n, "response": response}
            )

    output_dir = config["file_paths"]["generation"]
    output_file = os.path.join(output_dir, f"{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, output_file)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")'''

if __name__ == "__main__":
    main()
