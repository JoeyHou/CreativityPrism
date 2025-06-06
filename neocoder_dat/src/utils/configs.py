"""This module contains the configurations for getting
inputs for the models.
"""
from typing import Text, Dict, Any, List, Union, Optional, Iterable
from tqdm import tqdm
from collections import Counter
import json
import pandas as pd
import os
from torch.utils.data import DataLoader

from src.models.model import (
    OpenAIModel,
    AnthropicModel,
    OpenModel,
    OpenModelVLLM,
    OpenModelHF,
    GenAIModel,
    DeepSeekModel
)
from src.collate_fns.codeforce_collate_fn import CodeforceDPInferenceCollateFn

from src.generator.dataset import DPInferenceDataset

from src.generator.dp_generator import (
    APIModelParallelThreadDPInference,
    OpenModelParallelThreadDPInference
)

from src.generator.p_generator import (
    APIModelSingleInference,
    OpenModelSingleInference
)

from src.evaluators.dp_evaluator import CodeForceCorrectnessEvaluator

from src.generator.generator import TECHNIQUES

__MODEL_TO_CLASS__ = {
    "gpt-4-turbo": OpenAIModel,
    "gpt-4.1": OpenAIModel,
    "gpt-4.1-mini": OpenAIModel,
    "gpt-4": OpenAIModel,
    "gpt-3.5-turbo": OpenAIModel,
    "gemini-2.5-pro-preview-03-25": GenAIModel,
    "gemini-2.0-flash": GenAIModel,
    "gemini-1.5-pro": GenAIModel,
    "deepseek-chat": DeepSeekModel,
    "deepseek-reasoner": DeepSeekModel,
    # "anthropic.claude-3-sonnet-20240229-v1:0": AnthropicModel,
    "claude-3-5-haiku-20241022": AnthropicModel,
    "claude-3-7-sonnet-20250219": AnthropicModel,
    "meta-llama/Llama-2-70b-chat-hf": OpenModelVLLM,
    "meta-llama/Llama-2-13b-chat-hf": OpenModelVLLM,
    "meta-llama/Llama-2-7b-chat-hf": OpenModelVLLM,
    "meta-llama/Meta-Llama-3-70B-Instruct": OpenModelHF,
    "meta-llama/Llama-3.1-8B": OpenModelVLLM,
    "meta-llama/Llama-3.1-8B-Instruct": OpenModelVLLM,
    "allenai/tulu-2-7b": OpenModelVLLM,
    "allenai/tulu-2-13b": OpenModelVLLM,
    "allenai/tulu-2-70b": OpenModelVLLM,
    "Salesforce/codet5p-6b": OpenModelHF,
    "Salesforce/codet5p-16b": OpenModelHF,
    "Salesforce/instructcodet5p-16b": OpenModelHF,
    "allenai/unifiedqa-t5-11b": OpenModelHF,
    "Xwin-LM/XwinCoder-7B": OpenModelVLLM,
    "Xwin-LM/XwinCoder-13B": OpenModelVLLM,
    "Xwin-LM/XwinCoder-34B": OpenModelVLLM,
    "Xwin-LM/Xwin-LM-70B-V0.1": OpenModelVLLM,
    "codellama/CodeLlama-34b-Instruct-hf": OpenModelHF, #  Code completion. / Instruction and chat
    "codellama/CodeLlama-34b-Python-hf": OpenModelHF,   #  Code completion. / Python specialist
    "codellama/CodeLlama-34b-hf": OpenModelHF,          #  Code completion
    "bigcode/starcoder": OpenModelVLLM,                 #  StarCoderbase trained on additional 30B tokens of Python
    "bigcode/starcoderplus": OpenModelVLLM,               
    "WizardLM/WizardCoder-Python-34B-V1.0": OpenModelVLLM,
    "google/codegemma-7b-it": OpenModelHF,
    "mistralai/Mistral-7B-v0.1": OpenModelVLLM,
    "mistralai/Mistral-7B-Instruct-v0.3": OpenModelVLLM,
    "Qwen/Qwen2.5-7B-Instruct": OpenModelVLLM,
    "Qwen/Qwen2.5-Coder-7B-Instruct": OpenModelVLLM,
    "allenai/OLMo-2-1124-7B-Instruct": OpenModelVLLM,
    "allenai/OLMo-2-1124-13B-Instruct": OpenModelVLLM,
    "allenai/OLMo-2-1124-13B-DPO": OpenModelVLLM,
    "allenai/OLMo-2-1124-13B-SFT": OpenModelVLLM,
}

CODEFORCE_GENERATOR = """You are a Python code generator, only return the import and python function. Input will be an very detailed description of task, output will be the code.
The input will be from command line, and the output will be printed to the console as well. Your result will be solely a function named solve(), and do not call this function in your code.
Make sure the code is free of bug and can pass the test cases provided. You can use any library you want. The test cases are provided in the code. Do not call the solve() function in your code."""
DAT_PROMPT="""Please write 10 nouns in English that are as irrelevantfrom each other as possible, in all meanings and uses of the words. Please note that the words you write should have only single word, only nouns (e.g., things, objects, concepts), and no proper nouns (e.g., no specific people or places). Your answer:"""


__SYS_PROMPT__ = {
    "codeforce": CODEFORCE_GENERATOR
}

CODE_REVIEWER = """You are a code reviewer. Detect all the programming techniques from the input and return a list of programming techniques. Only select the techniques from this list: """ + \
f'{TECHNIQUES}' + \
"""\nYour output should look like this:\n- technique 1\n- technique 2\n- technique 3\n- ..."""


def get_inference_params(
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int
):

    try:
        model_class = __MODEL_TO_CLASS__[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} is not supported.")

    is_open_model = model_class != OpenAIModel and model_class != AnthropicModel and model_class != GenAIModel and model_class != DeepSeekModel
    use_vllm = model_class == OpenModelVLLM    # certain open-source models are not supported by VLLM

    if is_open_model:
        model: OpenModel = model_class(model_name=model_name, prompt=None)
    else:
        model: Union[OpenAIModel, AnthropicModel, GenAIModel, DeepSeekModel] =  model_class(model=model_name,
                                                                                            temperature=temperature,
                                                                                            top_p=top_p,
                                                                                            max_tokens=max_tokens)

    open_model_config = {
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,   
        "num_return_sequences": 1,
        "num_beam_groups": 1,
        "num_beams": 1,
    } if not use_vllm else {
        "max_tokens": max_tokens,
        'temperature': temperature,
        "top_p": top_p,
        'n': 1
    }

    generator = OpenModelSingleInference(
        model=model,
        config=open_model_config,
        use_vllm=use_vllm
    ) if is_open_model else \
                APIModelSingleInference(
        model=model
    ) 

    return {
        "generator": generator
    }

def get_dp_inference_params(
    dataset_dir: str,
    model_name: str,
    dp_rounds: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int
):
    """Get the parameters for parallel thread DP inference. 
    """

    if "codeforce" in dataset_dir.lower():
        dataset_name = "codeforce"
    else:
        raise ValueError(f"Dataset {dataset_dir} is not supported.")

    dataset = DPInferenceDataset(dataset_dir,
                                 dp_rounds=dp_rounds)

    try:
        model_class = __MODEL_TO_CLASS__[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} is not supported.")

    is_open_model = model_class != OpenAIModel and model_class != AnthropicModel and model_class != GenAIModel and model_class != DeepSeekModel
    use_vllm = model_class == OpenModelVLLM    # certain open-source models are not supported by VLLM

    if is_open_model:
        model: OpenModel = model_class(model_name=model_name,
                                       prompt=__SYS_PROMPT__[dataset_name])
        
        collate_fn = CodeforceDPInferenceCollateFn(
            model_name=model_name,
            tokenizer=model.tokenizer,
            is_open_model=is_open_model,
            use_vllm=use_vllm,
            dp_rounds=dp_rounds,
            prompt=model.prompt
        )

    else:
        model: Union[OpenAIModel, AnthropicModel, GenAIModel, DeepSeekModel] =  model_class(model=model_name, 
                                                                             gpt_setting=__SYS_PROMPT__[dataset_name],
                                                                             temperature=temperature,
                                                                             top_p=top_p,
                                                                             max_tokens=max_tokens)
        
        collate_fn = CodeforceDPInferenceCollateFn(
            model_name=model_name,
            tokenizer=None,
            is_open_model=False,
            use_vllm=False,
            dp_rounds=dp_rounds,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size if is_open_model else 1,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # sampling config for Open-Source models
    open_model_config = {
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,   
        "num_return_sequences": 1,
        "num_beam_groups": 1,
        "num_beams": 1,
    } if not use_vllm else {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1
    }

    generator = OpenModelParallelThreadDPInference(
        model=model,
        dp_rounds=dp_rounds,
        config=open_model_config,
        use_vllm=use_vllm
    ) if is_open_model else \
                APIModelParallelThreadDPInference(
        model=model,
        dp_rounds=dp_rounds,
    ) 

    return {
        "generator": generator,
        "dataloader": dataloader,
    }


def correctness_evaluation(inference_result_path: str,
                           test_case_path: str,
                           save_folder: str):
    
    os.makedirs(save_folder, exist_ok=True)

    codeforce_correctness_evaluator = CodeForceCorrectnessEvaluator(inference_result_path, 
                                                          test_case_path)
    correctness = codeforce_correctness_evaluator.evaluate()
    model_name = codeforce_correctness_evaluator.model_name
    num_sample = codeforce_correctness_evaluator.num_sample
    num_dp = codeforce_correctness_evaluator.num_dp
    out_file = os.path.join(save_folder, 
                         f"{model_name}_sample={num_sample}_dp={num_dp}_creativity.json")
    
    with open(out_file, "w") as f:
        json.dump(correctness, f, indent=4)


def technique_detection(human_solution_path: str,
                        inference_result_path: str):
    """Detect the programming techniques in human solutions and the generated codes.
    Only applicable to Codeforce dataset
    """

    with open(inference_result_path, "r") as f:
        inference_result = json.load(f)
    with open(human_solution_path, "r") as f:
        human = json.load(f)
    
    save_path = os.path.join(os.path.dirname(human_solution_path), "human_solution_techniques.json")
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            human_tech = json.load(f)
    else:
        human_tech = {}

    code_reviewer = OpenAIModel(model="gpt-4-turbo", temperature=0.75, top_p=1, max_tokens=256, gpt_setting=CODE_REVIEWER)
    code_evaluator = CodeForceCorrectnessEvaluator(inference_result_path, test_case_path=None) # only used for parsing codes
    code_generator = APIModelParallelThreadDPInference("gpt-4-turbo", dp_rounds=0) # only used for parsing techniques
    
    def _detect_techniques(code: Text) -> List[Text]:
        while True:
            technique_str = code_reviewer(code)[0]
            code_reviewer.restart()
            techniques = code_generator.parse_techniques(technique_str)
            # sometimes GPT-4 just repeats all techniques in the prompt
            # in such case we let the model to re-generate the techniques
            if len(techniques) <= 10:
                break

        return techniques
    
    def save_json(data: Dict[str, Any], save_path: str):
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    for problem in tqdm(inference_result, desc="Detecting techniques"):
        problem_id = problem["problem_id"]
        human_solutions: List[Text] = human[problem_id]
        if "codes" in problem:
            model_generated_codes = []
            for idx, output in enumerate(problem['codes']):
                if output is not None:
                    model_generated_codes.append(code_evaluator.parse_code(output))
                else:
                    # sometimes the OpenAI API returns code not in the ```python ... ``` format
                    model_generated_codes.append(code_evaluator.parse_code(problem['outputs'][idx]))
        else:
            model_generated_codes: List[Text] = [code_evaluator.parse_code(code) for code in problem['outputs']]

        # Detect techniques in human solutions
        if problem_id in human_tech and len(human_tech[problem_id]) == len(human_solutions):
            pass
        else:
            techniques = []
            for human_solution in human_solutions:
                techniques.append(_detect_techniques(human_solution)) if human_solution else techniques.append([])
            
            human_tech.update({problem_id: techniques})
            save_json(human_tech, save_path)

        # Detect techniques in model generated codes
        if "techniques" in problem and len(problem["techniques"]) == len(model_generated_codes) and all(len(tech) <= 10 for tech in problem["techniques"]):
            pass
        else:
            techniques = []
            for model_generated_code in model_generated_codes:
                techniques.append(_detect_techniques(model_generated_code)) if model_generated_code else techniques.append([])
            
            problem.update({"techniques": techniques})
            save_json(inference_result, inference_result_path)

        # monitor OpenAI API usage
        print(code_reviewer.gpt_usage(model = "gpt-4-turbo"))

def calculate_creativity(inference_result_path: str,
                         human_solution_path: str,
                         save_folder: str,
                         dp_rounds: int = 5):
    """Calculate the final creativity score
    """
    with open(inference_result_path, "r") as f:
        model_solutions = json.load(f)

    with open(human_solution_path, "r") as f:
        human = json.load(f)
    
    assert "correctness" in model_solutions[-1], "Please run correctness evaluation before calculating creativity."
    assert "techniques" in model_solutions[-1], "Please run technique detection before calculating creativity."

    human_solutions = {k: [t for ts in v for t in ts] for k, v in human.items()}
    human_solutions_counter = {k: Counter(v) for k, v in human_solutions.items()}
    # get the size of each human solutions counter
    human_solutions_size = {k: len(v.values()) for k, v in human_solutions_counter.items()}
    # sort the human_solution_size
    human_solutions_size = dict(sorted(human_solutions_size.items(), key=lambda x: x[1], reverse=True))

    results = dict(problem_id = [], 
                   dp = [],
                   constraints = [], 
                   machine_techniques = [], 
                   human_techniques = [], 
                   machine_solutions = [], 
                   correctness = [])
    
    code_evaluator = CodeForceCorrectnessEvaluator(inference_result_path, test_case_path=None) # only used for parsing codes
    
    for problem in model_solutions:
        problem_id = problem["problem_id"]
        if "codes" in problem:
            model_codes = []
            for idx, output in enumerate(problem['codes']):
                if output is not None: 
                    model_codes.append(code_evaluator.parse_code(output))
                else:
                    # sometimes the OpenAI API returns code not in the ```python ... ``` format
                    model_codes.append(code_evaluator.parse_code(problem['outputs'][idx]))
        elif "outputs" in problem:
            model_codes = [code_evaluator.parse_code(output) for output in problem["outputs"]]
        else:
            raise ValueError("No codes or outputs found in model solutions")

        if "constraints" in problem:
            constraints = problem["constraints"]
        elif "constraints_list" in problem:
            constraints = problem["constraints_list"]
        else:
            raise ValueError("No constraints found in model solutions")

        human_technique = list(human_solutions_counter[problem_id].keys())
        dp_idx = 0
        prev_constraint = None
        for constraint, model_technique, model_code, correctness in zip(constraints, problem["techniques"], model_codes, problem['correctness']):

            if constraint == prev_constraint:
                continue
            else:
                results["problem_id"].append(problem_id)
                results["dp"].append(dp_idx)
                results["constraints"].append(constraint)
                results["machine_techniques"].append(model_technique)
                results["human_techniques"].append(human_technique)
                results["machine_solutions"].append(model_code)
                results["correctness"].append(correctness)
                dp_idx += 1
                prev_constraint = constraint

    results = pd.DataFrame(results)
    results.set_index('problem_id', inplace=True)

    def check_constraints(row):
        return not bool(set(row["machine_techniques"]) & set(row["constraints"]))
    
    def check_techniques(row):
        if row["machine_techniques"] == []:
            return 0
        else:
            return len(set(row["machine_techniques"]) - set(row["human_techniques"]))

    def calcualte_new_techniques_ratio(row):
        if row["machine_techniques"] == []:
            return 0
        else:
            return row['new_techniques'] / len(row['machine_techniques'])

    results["follow_constraints"] = results.apply(check_constraints, axis=1)
    results["new_techniques"] = results.apply(check_techniques, axis=1)
    results["new_techniques_ratio"] = results.apply(calcualte_new_techniques_ratio, axis=1)

    # delete rows 1773F, as we cannot crawl its human solutions due to the website's restriction
    results = results[results.index != "1773F"]

    def calculate_convergent_thinking(results, dp_rounds):
        """probability of following constraints and correctness at dp_rounds
        """

        num_samples = len(results[results["dp"] == dp_rounds])
        num_correct_samples = len(results[(results["dp"] == dp_rounds) & (results["follow_constraints"] == True) & (results["correctness"] == True)])
        
        return num_correct_samples / num_samples

    def calculaate_divergent_thinking(results, dp_rounds):
        """Average number of new techniques at dp_rounds
        """

        return results[results["dp"] == dp_rounds]["new_techniques_ratio"].mean()
    
    def calculate_creativity(results, dp_rounds):
        """Probability of convergent thinking and divergent thinking at dp_rounds
        """

        dp_cluster = results[results["dp"] == dp_rounds]
        num_creative_samples = dp_cluster.apply(lambda x: x["follow_constraints"] * x["correctness"] * x["new_techniques_ratio"], axis=1).sum()
        return num_creative_samples / len(dp_cluster)

    convergent_thinking = [calculate_convergent_thinking(results, i) for i in range(0, dp_rounds+1)]
    divergent_thinking = [calculaate_divergent_thinking(results, i) for i in range(0, dp_rounds+1)]
    creativity = [calculate_creativity(results, i) for i in range(0, dp_rounds+1)]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    base_name = os.path.basename(inference_result_path).split("_sample")[0] 
    # save the results for further analysis
    results.to_csv(os.path.join(save_folder, base_name + "_creativity.csv"))

    return convergent_thinking, divergent_thinking, creativity
