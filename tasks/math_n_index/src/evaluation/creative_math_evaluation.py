import argparse
import logging
from datetime import datetime
import os
from tqdm import tqdm
import json
from src.models import ModelWrapper, ModelWrapperVLLM
from src.prompt_engineering import (load_coarse_grained_novelty_evaluation_prompt,
                      load_correctness_evaluation_prompt,
                      load_fine_grained_novelty_evaluation_prompt)

from src.utils import extract_yes_no, load_json, save_json, setup_logger

def load_config(input_file):
    file_path = input_file
    with open(file_path, "r") as file:
        return json.load(file)

config = load_config(input_file="configs/eval_creative_math.json")

save_interval = 5  # Save results after every 20 samples
# TODO: what model do we need for eval from now? 
# haven't been done
# eval_lm = ["Qwen2.5-7B-Instruct", "Llama-31-8B-Instruct"]


def evaluation(model_to_evaluate, args, logger, eval_lm):

    # Load data and generated results
    data_path = config["file_paths"]["dataset"]
    # corrected load path
    generation_path = os.path.join(config["file_paths"]["generation"], model_to_evaluate, f"{model_to_evaluate}.json")
    evaluation_dir = config["file_paths"]["evaluation"]
    # use this to cache the file to be loaded again
    evaluation_path = os.path.join(config["file_paths"]["evaluation"], f"{model_to_evaluate}_qwen.json")
    os.makedirs(evaluation_dir, exist_ok=True)

    data = load_json(data_path)
    amount_used = int(len(data) * args.portion)
    data = data[:amount_used]

    # If evaluation file exists, load it; otherwise, initialize evaluation results.
    if os.path.exists(evaluation_path):
        results = load_json(evaluation_path)
    else:
        results = load_json(generation_path)
        for sample in results:
            sample["correctness"] = {}
            sample["reasons"] = {
                "correctness": [],
                "coarse-grained": {},
                "fine-grained": {}
            }
            sample["solution_provided"] = {}
            sample["coarse_grained_novelty"] = {}
            sample["fine_grained_novelty"] = {}
    amount_used_result = int(len(results) * args.portion)
    results = results[:amount_used_result]
    # print(len(results))
    # step 1: correctness
    
    for eval_model in eval_lm:
        logger.info(f"Starting Correctness Evaluation with {eval_model} ...")
        evaluator_model = ModelWrapperVLLM(eval_model)
        prompts = []
        indices = []  # Record which sample each prompt belongs to

        for idx, sample in enumerate(results):
            # Skip if already evaluated for this model
            if eval_model in sample["correctness"]:
                continue
            if "final_decision" in sample["correctness"]:
                print("Skip!")
                continue

            q_idx = sample["question_number"]  # using question_number as index in data
            problem = sample["problem"] # data[q_idx]["input"]["text"]
            assert problem == data[q_idx]["input"]["text"]
            if sample["problem_id"] != data[q_idx]["meta_data"]["id"]:
                print(sample["problem_id"], " ", data[q_idx]["meta_data"]["id"])
                assert sample["problem_id"] == data[q_idx]["meta_data"]["id"]
            solutions = data[q_idx]["input"]["others"]["references"]["solutions"]
            new_solution = sample["cleaned_response"]
            prompt = load_correctness_evaluation_prompt(problem, solutions, new_solution)
            prompts.append(prompt)
            indices.append(idx)
            # sample["solution_provided"]["one"] = solutions # TODO: remove later
        if prompts:
            responses = evaluator_model.generate_batch_response(prompts)
            for i, resp in zip(indices, responses):
                decision = extract_yes_no(resp)
                results[i]["correctness"][eval_model] = decision
                results[i]["reasons"]["correctness"].append({eval_model: resp})
        # save results periodically
        if (idx + 1) % save_interval == 0:
            save_json(results, evaluation_path)
        logger.info(f"Finished Correctness Evaluation for {eval_model}")

    # Determine final correctness decision per sample: all evaluators must say "YES"
    for idx, sample in enumerate(results):
        all_yes = all(value == "YES" for value in sample["correctness"].values())
        sample["correctness"]["final_decision"] = "YES" if all_yes else "NO"
        results[idx] = sample
    save_json(results, evaluation_path)
    
    # step2 : coarse grained
    for eval_model in eval_lm:
        logger.info(f"Starting Coarse-Grained Novelty Evaluation with {eval_model} ...")
        evaluator_model = ModelWrapperVLLM(eval_model)
        prompts = []
        indices = []

        for idx, sample in enumerate(results):
            if eval_model in sample["coarse_grained_novelty"]:
                continue

            # Only evaluate correct solutions; otherwise, set as "NO"
            if sample["correctness"]["final_decision"] == "NO":
                sample["coarse_grained_novelty"][eval_model] = "NO"
                continue

            q_idx = sample["question_number"]
            problem = data[q_idx]["input"]["text"]
            solutions = data[q_idx]["input"]["others"]["references"]["solutions"]
            new_solution = sample["cleaned_response"]
            k = sample["k"]
            prompt = load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution)
            prompts.append(prompt)
            indices.append(idx)
        if prompts:
            responses = evaluator_model.generate_batch_response(prompts)
            for i, resp in zip(indices, responses):
                decision = extract_yes_no(resp)
                results[i]["coarse_grained_novelty"][eval_model] = decision
                results[i]["reasons"]["coarse-grained"] = resp
        # save results periodically
        if (idx + 1) % save_interval == 0:
            save_json(results, evaluation_path)
        logger.info(f"Finished Coarse-Grained Novelty Evaluation for {eval_model}")
    
    # Determine final decision by majority voting for coarse-grained novelty
    for idx, sample in enumerate(results):
        yes_count = sum(1 for value in sample["coarse_grained_novelty"].values() if value == "YES")
        no_count = sum(1 for value in sample["coarse_grained_novelty"].values() if value == "NO")
        sample["coarse_grained_novelty"]["final_decision"] = "YES" if yes_count > no_count else "NO"
        results[idx] = sample
    save_json(results, evaluation_path)

    # step 3: fine-grained
    for eval_model in eval_lm:
        logger.info(f"Starting Fine-Grained Novelty Evaluation with {eval_model} ...")
        evaluator_model = ModelWrapperVLLM(eval_model)
        prompts = []
        indices = []

        for idx, sample in enumerate(results):
            if eval_model in sample["fine_grained_novelty"]:
                continue

            # Only evaluate if the coarse novelty is positive and k < n
            if (sample["coarse_grained_novelty"]["final_decision"] == "NO") or (sample["k"] == sample["n"]):
                sample["fine_grained_novelty"][eval_model] = "NO"
                continue

            q_idx = sample["question_number"] # note the question_number is same as question_id, is unique for each question
            problem = data[q_idx]["input"]["text"]
            solutions = data[q_idx]["input"]["others"]["references"]["solutions"]
            new_solution = sample["response"]
            k = sample["k"]
            prompt = load_fine_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution)
            prompts.append(prompt)
            indices.append(idx)
            sample["solution_provided"]["three"] = solutions # TODO: remove later
        if prompts:
            responses = evaluator_model.generate_batch_response(prompts)
            for i, resp in zip(indices, responses):
                decision = extract_yes_no(resp)
                results[i]["fine_grained_novelty"][eval_model] = decision
                results[i]["reasons"]["fine-grained"] = resp

        # save results periodically
        if (idx + 1) % save_interval == 0:
            save_json(results, evaluation_path)
        logger.info(f"Finished Fine-Grained Novelty Evaluation for {eval_model}")

    # Determine final decision by majority voting for fine-grained novelty
    for idx, sample in enumerate(results):
        yes_count = sum(1 for value in sample["fine_grained_novelty"].values() if value == "YES")
        no_count = sum(1 for value in sample["fine_grained_novelty"].values() if value == "NO")
        sample["fine_grained_novelty"]["final_decision"] = "YES" if yes_count > no_count else "NO"
        results[idx] = sample
    # results = sorted(results, key=lambda x: (x["question_number"], x["k"])) # used for better format in the output file
    save_json(results, evaluation_path)
    
    '''
def main():
    parser = argparse.ArgumentParser(description="Run the evaluation program.")
    parser.add_argument(
        "--model_to_evaluate",
        type=str,
        default="Llama-31-8B-instruct",
        help="The model used in the experiment to be evaluated.",
    )
    parser.add_argument(
        "--portion",
        type=float,
        default=1,
        help="The fraction of data to use (between 0 and 1)."
    )
    parser.add_argument(
        "--eval_lm",
        type=str,
        default="Llama-3.3-70B",
        help="The judge model.",
    )
    args = parser.parse_args()
    model_to_evaluate = args.model_to_evaluate
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # use daytime to trace logfiles for final eval
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
    
    evaluation(model_to_evaluate = model_to_evaluate, args = args, logger=logger, eval_lm=[args.eval_lm])
    # Calculate accuracy metrics after all evaluations are done
    results = load_json(os.path.join(config["file_paths"]["evaluation"], f"{model_to_evaluate}_qwenllama.json"))
    N = len(results)
    # Ensure consistency in evaluation results
    for sample in results:
        if sample["correctness"]["final_decision"] == "NO":
            sample["coarse_grained_novelty"]["final_decision"] = "NO"
            sample["fine_grained_novelty"]["final_decision"] = "NO"
        elif sample["coarse_grained_novelty"]["final_decision"] == "NO":
            sample["fine_grained_novelty"]["final_decision"] = "NO"

    # Compute counts
    correctness_count = sum(1 for sample in results if sample["correctness"]["final_decision"] == "YES")
    coarse_count = sum(1 for sample in results if sample["coarse_grained_novelty"]["final_decision"] == "YES")
    fine_count = sum(1 for sample in results if sample["fine_grained_novelty"]["final_decision"] == "YES")

    # Compute ratios
    correctness_ratio = correctness_count / N
    novelty_ratio = coarse_count / N
    novel_unknown_ratio = fine_count / N

    # Compute conditional ratios with safety checks
    novelty_to_correctness_ratio = coarse_count / correctness_count if correctness_count else 0
    novel_unknown_to_novelty_ratio = fine_count / coarse_count if coarse_count else 0

    # Log results
    # only look at the last result which would be the joint eval from all judges
    logger.info(f"The evaluation result for {model_to_evaluate} is as follows:")
    logger.info(f"Correctness Ratio: {correctness_ratio:.2%}")
    logger.info(f"Novelty Ratio: {novelty_ratio:.2%}")
    logger.info(f"Novel-Unknown Ratio: {novel_unknown_ratio:.2%}")
    logger.info(f"Novelty-to-Correctness Ratio: {novelty_to_correctness_ratio:.2%}")
    logger.info(f"Novel-Unknown-to-Novelty Ratio: {novel_unknown_to_novelty_ratio:.2%}")
    '''
def main():
    parser = argparse.ArgumentParser(description="Run the evaluation program.")
    parser.add_argument(
        "--portion",
        type=float,
        default=1,
        help="The fraction of data to use (between 0 and 1)."
    )
    parser.add_argument(
        "--eval_lm",
        type=str,
        default="Llama-3.3-70B",
        help="The judge model.",
    )
    args = parser.parse_args()

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for logs
    log_file = os.path.join(config["logging"]["log_dir"], f"evaluation_{timestamp}.log")
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting batch evaluation of models...")

    # Extract models to evaluate from the config file
    # revise back later to include mistral 24B
    models_to_evaluate = [exp["model_name"] for i, exp in enumerate(config["experiments_list"])][0:1]

    logger.info(f"Models to evaluate: {models_to_evaluate}")
    
    for model_to_evaluate in models_to_evaluate:
        logger.info(f"Evaluating model: {model_to_evaluate}")
        
        # Run evaluation for the current model
        evaluation(model_to_evaluate=model_to_evaluate, args=args, logger=logger, eval_lm=[args.eval_lm])

        # After each model, compute and log evaluation metrics
        evaluation_path = os.path.join(config["file_paths"]["evaluation"], f"{model_to_evaluate}_qwen.json")
        if os.path.exists(evaluation_path):
            results = load_json(evaluation_path)
            N = len(results)

            # Ensure consistency in evaluation results
            for sample in results:
                if sample["correctness"]["final_decision"] == "NO":
                    sample["coarse_grained_novelty"]["final_decision"] = "NO"
                    sample["fine_grained_novelty"]["final_decision"] = "NO"
                '''
                elif sample["coarse_grained_novelty"]["final_decision"] == "NO":
                    sample["fine_grained_novelty"]["final_decision"] = "NO"
                '''
            # Compute counts
            correctness_count = sum(1 for sample in results if sample["correctness"]["final_decision"] == "YES")
            coarse_count = sum(1 for sample in results if sample["coarse_grained_novelty"]["final_decision"] == "YES")
            # fine_count = sum(1 for sample in results if sample["fine_grained_novelty"]["final_decision"] == "YES")

            # Compute ratios
            correctness_ratio = correctness_count / N if N else 0
            novelty_ratio = coarse_count / N if N else 0
            # novel_unknown_ratio = fine_count / N if N else 0

            # Compute conditional ratios with safety checks
            novelty_to_correctness_ratio = coarse_count / correctness_count if correctness_count else 0
            # novel_unknown_to_novelty_ratio = fine_count / coarse_count if coarse_count else 0

            # Log results
            logger.info(f"Evaluation result for {model_to_evaluate}:")
            logger.info(f"Correctness Ratio: {correctness_ratio:.2%}")
            with open("evaluation_results_creative_math_cleaned_data_April25.txt", "a") as f:
                f.write(f"Evaluator: {args.eval_lm}")
                f.write(f"Evaluation result for {model_to_evaluate}:\n")
                f.write(f"Correctness Ratio: {correctness_ratio:.2%}\n\n")

            logger.info(f"Novelty Ratio: {novelty_ratio:.2%}")
            # logger.info(f"Novel-Unknown Ratio: {novel_unknown_ratio:.2%}")
            logger.info(f"Novelty-to-Correctness Ratio: {novelty_to_correctness_ratio:.2%}")
            # logger.info(f"Novel-Unknown-to-Novelty Ratio: {novel_unknown_to_novelty_ratio:.2%}")

    logger.info("All models evaluated successfully!")


if __name__ == "__main__":
    main()