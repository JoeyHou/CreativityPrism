import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

# TODO: we first keep the original implementation, shall we remove this?
from src.models import ModelWrapper, ModelWrapperVLLM
from src.prompt_engineering.creative_math_prompts import load_novel_solution_generation_prompt
from src.utils import load_json, save_json, setup_logger

def load_config(input_file):
    file_path = input_file
    with open(file_path, "r") as file:
        return json.load(file)

config = load_config(input_file="/scratch/dkhasha1/bzhang90/creative_bench/configs/creative_math_config.json")

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
        default=1,
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
    ''' previously was not batched input
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
    '''
    
    # Phase 1: Accumulate all prompts and metadata for batch processing.
    all_prompts = []
    all_metadata = []
    for question_number, sample in tqdm(enumerate(data), desc="Preparing prompts"):
        problem_id = sample["meta_data"]["id"]
        dataset_name = sample["meta_data"]["dataset"]
        problem = sample["input"]["text"]
        solutions = sample["input"]["others"]["references"]["solutions"]
        n = len(solutions)
    
        # For each possible number of reference solutions (k from 1 to n)
        for k in range(1, n + 1):
            prompt = load_novel_solution_generation_prompt(problem, solutions, k)
            all_prompts.append(prompt)
            all_metadata.append({
                "problem": problem,
                "problem_id": problem_id,
                "question_number": question_number,
                "dataset": dataset_name,
                "k": k,  # current number of reference solutions used
                "n": n,  # total number of reference solutions available
                "ground_truth_solutions": solutions[:k] # added for easier mauual check
            })

    # Phase 2: Run batched inference using the model's batched method.
    responses = model.generate_batch_response(all_prompts)

    # Phase 3: Reassemble the results by merging metadata and responses.
    results = []
    for meta, response in zip(all_metadata, responses):
        meta["response"] = response
        results.append(meta)

    # Phase 4: Save the aggregated results.
    output_dir = config["file_paths"]["generation"]
    output_file = os.path.join(output_dir, f"{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, output_file)
    
    logger.info(f"Results saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
