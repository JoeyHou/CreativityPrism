import argparse
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm

from api_eval_math import CorrectnessEvaluator, CoarseNoveltyEvaluator, FineNoveltyEvaluator
from src.utils import load_json, save_json, setup_logger

def load_config(input_file):
    file_path = input_file
    with open(file_path, "r") as file:
        return json.load(file)

config = load_config(input_file="configs/eval_creative_math.json")
save_interval = 5

# define API keys for each provider
API_KEYS = {
    "openai": "key",
    "anthropic": "key",
    "google": "key"
}
# define judge models
JUDGE_MODELS = {
    "gpt-4.1": "openai",
    "claude-3-7-sonnet-20250219": "anthropic",
    "gemini-2.0-flash": "google"
}

def get_api_key(model_name):
    """
    Retrieve the correct API key based on the model's provider.
    """
    provider = JUDGE_MODELS.get(model_name)
    return API_KEYS.get(provider)

def main():
    parser = argparse.ArgumentParser(description="Run API-based evaluation program.")
    # model to be evaluated, not the evaluator
    parser.add_argument("--model_to_evaluate", type=str, default="OLMo2-13B-dpo")
    parser.add_argument("--portion", type=float, default=1)
    args = parser.parse_args()
    model_to_evaluate = args.model_to_evaluate

    # setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["logging"]["log_dir"], f"generation_{model_to_evaluate}_api_{timestamp}.log")
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting API-based evaluation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    # Load data and generated results
    data_path = config["file_paths"]["dataset"]
    # corrected load path
    generation_path = os.path.join(config["file_paths"]["generation"], model_to_evaluate, f"{model_to_evaluate}.json")
    print(generation_path)
    evaluation_dir = config["file_paths"]["evaluation"]
    # use this to cache the file to be loaded again
    evaluation_path = os.path.join(config["file_paths"]["evaluation"], f"{model_to_evaluate}_Claude_correctness.json")
    print(evaluation_path)
    os.makedirs(evaluation_dir, exist_ok=True)

    data = load_json(data_path)

    if os.path.exists(evaluation_path):
        results = load_json(evaluation_path)
    else:
        results = load_json(generation_path)
        for sample in results:
            sample["correctness"] = {}
            sample["reasons"] = {
                "correctness": {},
                "coarse-grained": {},
                "fine-grained": {}
            }
            sample["solution_provided"] = {}
            sample["coarse_grained_novelty"] = {}
            sample["fine_grained_novelty"] = {}

    # create the three different kinds of evaluators for each model
    correctness_evaluators = { # only one model name is passed in per time
        model: CorrectnessEvaluator(get_api_key(model), [model])
        for model in JUDGE_MODELS.keys()
    }
    coarse_evaluators = {
        model: CoarseNoveltyEvaluator(get_api_key(model), [model])
        for model in JUDGE_MODELS.keys()
    }
    fine_evaluators = {
        model: FineNoveltyEvaluator(get_api_key(model), [model])
        for model in JUDGE_MODELS.keys()
    }
    amount_used = int(len(results) * args.portion)
    results = results[:amount_used]
    
    # task one: correctness evaluation
    # different pipeline by calling api-models for each sample directly instead of loop through models
    logger.info("Starting Correctness Evaluation...")
    for idx, sample in tqdm(enumerate(results), total=len(results)):
        if "final_decision" in sample["correctness"]:
            logger.info(f"Skipping the problem number{idx} since it has been evaluated before for correctness!")
            continue

        # Use sample["question_number"] to locate the corresponding question in the dataset.
        # TODO: fix using sample and data together
        q_idx = sample["question_number"]
        assert sample["problem_id"] == data[q_idx]["meta_data"]["id"]
        # TODO: revise this to decomplicated
        # problem = data[q_idx]["input"]["text"]
        assert data[q_idx]["input"]["text"] == sample['problem']
        problem = sample['problem']
        solutions = data[q_idx]["input"]["others"]["references"]["solutions"] # since we need all solutions
        new_solution = sample["cleaned_response"]
        # TODO: fix this context; reply: do not need k value for correctness eval
        context = {"solutions": solutions, "new_solution": new_solution}
        # Store individual evaluator decisions
        
        # evaluator_decisions = {model: evaluator.evaluate(problem, context) for model, evaluator in correctness_evaluators.items()}
        
        # Store individual evaluator decisions directly
        # get all models' decisions
        for model, evaluator in correctness_evaluators.items():
            sample["correctness"][model], sample["reasons"]["correctness"][model] = evaluator.evaluate(problem, context)

        # Compute final decision based on all evaluators, different for methods below
        sample["correctness"]["final_decision"] = "YES" if all(d == "YES" for d in sample["correctness"].values()) else "NO"
        results[idx] = sample

        if (idx + 1) % save_interval == 0:
            save_json(results, evaluation_path)
    save_json(results, evaluation_path)
    logger.info("Finished Correctness Evaluation.")
    
    # task two: coarse-grained novelty evaluation
    # different pipeline by calling api-models for each sample directly instead of loop through models
    logger.info("Starting Coarse-Grained Novelty Evaluation...")
    for idx, sample in tqdm(enumerate(results), total=len(results)):
        # TODO: this may cause problem, revise! But should be fine for API models
        if "final_decision" in sample["coarse_grained_novelty"]:
             logger.info(f"Skipping index {idx} since it has been evaluated.")
             continue

        if sample["correctness"]["final_decision"] == "NO":
            sample["coarse_grained_novelty"]["final_decision"] = "NO"
            logger.info(f"Skipping index {idx} since it is incorrect!")
            continue

        q_idx = sample["question_number"]
        assert sample["problem_id"] == data[q_idx]["meta_data"]["id"]
        problem = data[q_idx]["input"]["text"]
        assert data[q_idx]["input"]["text"] == sample['problem']
        solutions = data[q_idx]["input"]["others"]["references"]["solutions"]
        new_solution = sample["cleaned_response"]
        k = sample["k"]
        context = {"solutions": solutions, "new_solution": new_solution, "k": k}

        # same procedure
        # Store individual evaluator decisions directly
        decisions = []
        for model, evaluator in coarse_evaluators.items():
            sample["coarse_grained_novelty"][model], sample["reasons"]["coarse-grained"][model] = evaluator.evaluate(problem, context)
            decisions.append(sample["coarse_grained_novelty"][model])
        
        yes_count = sum(1 for d in decisions if d == "YES")
        no_count = len(decisions) - yes_count
        # majority voting, need to revise back later though final_decision currently is not important
        sample["coarse_grained_novelty"]["final_decision"] = "YES" if yes_count >= no_count else "NO"
        results[idx] = sample

        if (idx + 1) % save_interval == 0:
            save_json(results, evaluation_path)
    save_json(results, evaluation_path)
    logger.info("Finished Coarse-Grained Novelty Evaluation.")
    
    '''
    # task three: fine-grained novelty evaluation
    # different pipeline by calling api-models for each sample directly instead of loop through models
    logger.info("Starting Fine-Grained Novelty Evaluation...")
    for idx, sample in tqdm(enumerate(results), total=len(results)):
        # TODO: this may cause problem, revise!
        if "final_decision" in sample["fine_grained_novelty"]:
            continue

        if sample["coarse_grained_novelty"]["final_decision"] == "NO" or (sample["k"] == sample["n"]):
            sample["fine_grained_novelty"]["final_decision"] = "NO"
            continue

        q_idx = sample["question_number"]
        assert sample["problem_id"] == data[q_idx]["meta_data"]["id"] # ensure matching
        problem = data[q_idx]["input"]["text"]
        assert data[q_idx]["input"]["text"] == sample['problem']
        solutions = data[q_idx]["input"]["others"]["references"]["solutions"]
        new_solution = sample["response"]
        k = sample["k"]
        context = {"solutions": solutions, "new_solution": new_solution, "k": k}
        sample["solution_provided"]["three"] = solutions # TODO: remove later
        decisions = []
        for model, evaluator in fine_evaluators.items():
            sample["fine_grained_novelty"][model], sample["reasons"]["fine-grained"][model] = evaluator.evaluate(problem, context)
            decisions.append(sample["fine_grained_novelty"][model])
        yes_count = sum(1 for d in decisions if d == "YES")
        no_count = len(decisions) - yes_count
        # majority voting
        sample["fine_grained_novelty"]["final_decision"] = "YES" if yes_count >= no_count else "NO"
        results[idx] = sample

        if (idx + 1) % save_interval == 0:
            save_json(results, evaluation_path)
    save_json(results, evaluation_path)
    logger.info("Finished Fine-Grained Novelty Evaluation.")
    # Note, each evaluation built on the previous one so I need to make it sequential for saing api calls
    # instead of do all the evaluations and filter out those desirable evaluations.
    '''
    # aggregate results
    N = len(results) # all data
    correctness_count = sum(1 for sample in results if sample["correctness"]["final_decision"] == "YES")
    coarse_count = sum(1 for sample in results if sample["coarse_grained_novelty"]["final_decision"] == "YES")
    # fine_count = sum(1 for sample in results if sample["fine_grained_novelty"]["final_decision"] == "YES")
    correctness_ratio = correctness_count / N
    novelty_ratio = coarse_count / N
    # novel_unknown_ratio = fine_count / N
    novelty_to_correctness_ratio = coarse_count / correctness_count if correctness_count else 0
    # novel_unknown_to_novelty_ratio = fine_count / coarse_count if coarse_count else 0

    logger.info(f"Evaluation Results for {model_to_evaluate}:")
    logger.info(f"Correctness Ratio: {correctness_ratio:.2%}")
    logger.info(f"Novelty Ratio: {novelty_ratio:.2%}")
    # logger.info(f"Novel-Unknown Ratio: {novel_unknown_ratio:.2%}")
    logger.info(f"Novelty-to-Correctness Ratio: {novelty_to_correctness_ratio:.2%}")
    # logger.info(f"Novel-Unknown-to-Novelty Ratio: {novel_unknown_to_novelty_ratio:.2%}")
    '''
    result_entry = {
        "model": model_to_evaluate,
        "correctness_ratio": correctness_ratio,
        "novelty_ratio": novelty_ratio,
        "novelty_to_correctness_ratio": novelty_to_correctness_ratio,
    }

    with open("evaluation_results_all_models.jsonl", "a") as f:
        f.write(json.dumps(result_entry) + "\n")
    '''
if __name__ == "__main__":
    main()
