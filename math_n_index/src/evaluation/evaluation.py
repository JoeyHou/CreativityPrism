import argparse
import logging
import os
from datetime import datetime

from tqdm import tqdm

from configs import config
from src.models import ModelWrapper, ModelWrapperVLLM
from src.prompt_engineering import (load_coarse_grained_novelty_evaluation_prompt,
                      load_correctness_evaluation_prompt,
                      load_fine_grained_novelty_evaluation_prompt)

from src.utils import extract_yes_no, load_json, save_json, setup_logger

save_interval = config["experiment"][
    "save_interval"
]  # Save results after every 20 samples
# TODO: what model do we need for eval from now? 
# haven't been done
evaluators = ["Llama-3.1-8B"]


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation program.")
    parser.add_argument(
        "--model_to_evaluate",
        type=str,
        default="Llama-3.1-8B",
        help="The model was used in the experiment and will be evaluated.",
    )
    # added for testing
    parser.add_argument(
        "--portion",
        type=int,
        default=0.04,
        help="The amount of data to use."
    )
    args = parser.parse_args()
    model_to_evaluate = args.model_to_evaluate

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"],
        f"generation_{model_to_evaluate}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the evaluation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")
    logger.warning(
        "Ensure all transition sentences and justifications explaining the uniqueness of new solutions are "
        "manually removed to avoid influencing evaluator judgment.\n"
        "These sentences are usually at the beginning or ending of the response."
    )

    data_path = config["file_paths"]["dataset"]
    # get the generated file, see generation for format
    generation_path = os.path.join(
        config["file_paths"]["generation"], f"{model_to_evaluate}.json"
    )
    # save model to the eval dir with its name
    evaluation_dir = config["file_paths"]["evaluation"]
    evaluation_path = os.path.join(evaluation_dir, f"{model_to_evaluate}.json")
    # TODO: revise this part
    data = load_json(data_path)
    amount_used = int(len(data) * args.portion)
    data = data[:amount_used]

    # Evaluation file exists. Continuing unfinished evaluation.
    if os.path.exists(evaluation_path):
        results = load_json(evaluation_path) # a list
    # Create the evaluation file and copy the experiment results.
    else:
        os.makedirs(evaluation_dir, exist_ok=True)
        results = load_json(generation_path)
        # originally used sample_number but was confusing.
        # this create three fileds for each question 
        # question_id is still a number!
        for question_number, sample in enumerate(results):
            results[question_number]["correctness"] = {}
            results[question_number]["coarse_grained_novelty"] = {}
            results[question_number]["fine_grained_novelty"] = {}

    # Stage 1: Correctness Evaluation
    for model_name in evaluators:
        model = ModelWrapperVLLM(model_name)
        # go over all problems
        for question_number, sample in tqdm(enumerate(results)):
            # Skip if the evaluation result exists
            if model_name in sample["correctness"]:
                continue

            # Load problem and all solutions
            # use the number of question instead of real id
            question_number = sample["question_number"]
            # data here is a list, and since we use the same procedure in generation, this is true
            # TODO: consider to directly use the id field?
            problem = data[question_number]["input"]["text"] # changed
            solutions = data[question_number]["input"]["others"]["references"]["solutions"] # changed, already a list

            # Load the generated new solution
            new_solution = sample["response"] # this is fine

            prompt = load_correctness_evaluation_prompt(
                problem, solutions, new_solution
            )
            response = model.generate_response(prompt)
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            # save if correct for three eval models
            sample["correctness"][model_name] = decision
            results[question_number] = sample # update

            # Save every 20 samples
            
            if question_number % save_interval == 0:
                save_json(results, evaluation_path)
        save_json(results, evaluation_path)

    # A new solution is classified as correct if all three evalution results are "YES"
    for question_number, sample in enumerate(results):
        all_yes = all(value == "YES" for value in sample["correctness"].values())
        sample["correctness"]["final_decision"] = "YES" if all_yes else "NO"
        results[question_number] = sample # update this field, final decision
    save_json(results, evaluation_path)

    # Stage 2: Coarse-Grained Novelty Assessment
    for model_name in evaluators:
        model = ModelWrapperVLLM(model_name)

        for question_number, sample in tqdm(enumerate(results)):
            # Skip if the evaluation result exists
            if model_name in sample["coarse_grained_novelty"]:
                continue

            # Only correct solution will be evaluated.
            # Otherwise classify decision as "NO" directly.
            if sample["correctness"]["final_decision"] == "NO":
                sample["coarse_grained_novelty"][model_name] = "NO"
                results[question_number] = sample # update
                continue

            # Load problem and all solutions
            # revised
            question_number = sample["question_number"]
            problem = data[question_number]["input"]["text"] 
            solutions = data[question_number]["input"]["others"]["references"]["solutions"]

            # Load the generated new solution
            new_solution = sample["response"]

            k = sample["k"] # get num of ref sol
            prompt = load_coarse_grained_novelty_evaluation_prompt(
                problem, solutions, k, new_solution
            )
            response = model.generate_response(prompt)
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            sample["coarse_grained_novelty"][model_name] = decision
            results[question_number] = sample

            # Save every 20 samples
            if question_number % save_interval == 0:
                save_json(results, evaluation_path)
        save_json(results, evaluation_path)

    # Determine the final decision based on majority voting
    # NO change
    for question_number, sample in enumerate(results):
        yes_count = sum(
            1 for value in sample["coarse_grained_novelty"].values() if value == "YES"
        )
        no_count = sum(
            1 for value in sample["coarse_grained_novelty"].values() if value == "NO"
        )
        sample["coarse_grained_novelty"]["final_decision"] = (
            "YES" if yes_count > no_count else "NO"
        )
        results[question_number] = sample
    save_json(results, evaluation_path)

    # Stage 3: Fine-Grained Novelty Assessment
    for model_name in evaluators:
        model = ModelWrapperVLLM(model_name)

        for question_number, sample in tqdm(enumerate(results)):
            # Skip if the evaluation result exists
            if model_name in sample["fine_grained_novelty"]:
                continue

            # Only solutions that pass the fine-grained novelty assessment will be evaluated.
            # Only samples where k < n will be evaluated.
            # Otherwise, classify the decision as "NO" directly.
            if (sample["coarse_grained_novelty"]["final_decision"] == "NO") or (
                sample["k"] == sample["n"]
            ):
                sample["fine_grained_novelty"][model_name] = "NO"
                results[question_number] = sample
                continue

            # Load problem and all solutions
            # revised
            question_number = sample["question_number"]
            problem = data[question_number]["input"]["text"] 
            solutions = data[question_number]["input"]["others"]["references"]["solutions"]

            # Load the generated new solution
            new_solution = sample["response"]

            k = sample["k"] # this load solutions after the k-th ref
            prompt = load_fine_grained_novelty_evaluation_prompt(
                problem, solutions, k, new_solution
            )
            response = model.generate_response(prompt)
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            sample["fine_grained_novelty"][model_name] = decision
            results[question_number] = sample

            # Save every 20 samples
            if question_number % save_interval == 0:
                save_json(results, evaluation_path)
        save_json(results, evaluation_path)

    # Determine the final decision based on majority voting
    for question_number, sample in enumerate(results):
        yes_count = sum(
            1 for value in sample["fine_grained_novelty"].values() if value == "YES"
        )
        no_count = sum(
            1 for value in sample["fine_grained_novelty"].values() if value == "NO"
        )
        sample["fine_grained_novelty"]["final_decision"] = (
            "YES" if yes_count > no_count else "NO"
        )
        results[question_number] = sample # update
    save_json(results, evaluation_path)

    # Calculate accuarcy
    N = len(results)
    correctness_count = 0
    coarse_grained_novelty_count = 0
    fine_grained_novelty_count = 0

    for sample in results:
        if sample["correctness"]["final_decision"] == "YES":
            correctness_count += 1
        if sample["coarse_grained_novelty"]["final_decision"] == "YES":
            coarse_grained_novelty_count += 1
        if sample["fine_grained_novelty"]["final_decision"] == "YES":
            fine_grained_novelty_count += 1

    correctness_ratio = correctness_count / N
    novelty_ratio = coarse_grained_novelty_count / N
    novel_unknown_ratio = fine_grained_novelty_count / N
    if correctness_count != 0:
        novelty_to_correctness_ratio = coarse_grained_novelty_count / correctness_count
    else:
        novelty_to_correctness_ratio = 0
    if coarse_grained_novelty_count != 0:
        novel_unknown_to_novelty_ratio = (
            fine_grained_novelty_count / coarse_grained_novelty_count
        )
    else:
        novel_unknown_to_novelty_ratio = 0

    logger.info(f"The evaluation result for {model_to_evaluate} is as follows:")
    logger.info(f"Correctness Ratio: {correctness_ratio:.2%}")
    logger.info(f"Novelty Ratio: {novelty_ratio:.2%}")
    logger.info(f"Novel-Unknown Ratio: {novel_unknown_ratio:.2%}")
    logger.info(f"Novelty-to-Correctness Ratio: {novelty_to_correctness_ratio:.2%}")
    logger.info(f"Novel-Unknown-to-Novelty Ratio: {novel_unknown_to_novelty_ratio:.2%}")


if __name__ == "__main__":
    main()
