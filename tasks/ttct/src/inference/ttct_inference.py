"""Inference for creative questions for seven tasks
"""
import argparse
import logging
import os
from collections import Counter
from tqdm import tqdm

from src.utils.helpers import load_json, load_prompts_as_list, save_json
# from src.utils.run_api import run_api

def _load_hf_token(api_key_path: str) -> str:
    """Read HF token from credentials file (``hf`` key), fall back to env."""
    try:
        import json
        with open(api_key_path) as _f:
            return json.load(_f).get("hf", "")
    except Exception:
        return ""

_api_key_path_default = os.environ.get(
    "CREATIVITYPRISM_API_KEYS",
    "/ihome/xli/joh227/developer/creativityprism_git_workspace/api_keys.json"
)
if not os.environ.get("HF_TOKEN"):
    _hf = _load_hf_token(_api_key_path_default)
    if _hf:
        os.environ["HF_TOKEN"] = _hf
MAX_TOKENS = 3072
TOP_P = 1
TOP_K = 50
DEFAULT_SUBSET = [
    '1_unusual_uses', '5_common_problems', '4_situation', '6_improvement', '2_consequences'
]
def main(model_name, temp, cache_dir, subset, prompt_formats, api_key_path, data_path = 'data/processed/ttct.json', run_id="", num_samples=-1):

    logger = logging.getLogger(__name__)
    full_data = [item for item in load_json(data_path)]
    print(f'Number of items after filtering by subset {subset}: {len([item for item in full_data if item["meta_data"]["question_type"] in subset])}')
    
    # Get data for all 3 prompt types
    if prompt_formats == 'all':
        prompt_formats = ['basic', 'instructive', 'cot']
    else:
        prompt_formats = prompt_formats.split(',')
    print(f'Using prompt formats: {prompt_formats}')
    if 'basic' in prompt_formats:
        text_basic_list = load_prompts_as_list(data_path=data_path, prompt_type="text_basic", subset=subset)
    else:
        text_basic_list = ['SKIPPED'] * len(full_data)
    if 'instructive' in prompt_formats:
        text_instructive_list = load_prompts_as_list(data_path=data_path, prompt_type="text_instructive", subset=subset)
    else:
        text_instructive_list = ['SKIPPED'] * len(full_data)
    if 'cot' in prompt_formats:
        text_cot_list = load_prompts_as_list(data_path=data_path, prompt_type="text_cot", subset=subset)
    else:
        text_cot_list = ['SKIPPED'] * len(full_data)

    if num_samples > 0:
        # Evaluation asserts this file has one row per basefile.csv row, so cap each question
        # type by marking the surplus SKIPPED rather than truncating. The `skip` flag rides
        # through json2csv/csv2json so the judge does not score the capped rows either.
        kept = Counter()
        for i, item in enumerate(full_data):
            question_type = item['meta_data']['question_type']
            if subset and question_type not in subset:
                continue
            kept[question_type] += 1
            if kept[question_type] > num_samples:
                text_basic_list[i] = 'SKIPPED'
                text_instructive_list[i] = 'SKIPPED'
                text_cot_list[i] = 'SKIPPED'
                item['skip'] = True
        running = sum(min(c, num_samples) for c in kept.values())
        print(f'Limiting to the first {num_samples} item(s) of each of the {len(kept)} question '
              f'type(s): {running} of {len(full_data)} rows will be queried (smoke-test mode)')

    print('=> Non-skipped counts:')
    for prompt_format in ['basic', 'instructive', 'cot']:
        non_skipped_count = sum(1 for i in locals()[f'text_{prompt_format}_list'] if i != 'SKIPPED')
        print(f'Prompt Format: {prompt_format}, Non-skipped Count: {non_skipped_count}')

    if 'gpt' in model_name.lower() or 'claude' in model_name.lower() or 'gemini' in model_name.lower() or 'deepseek' in model_name.lower():
        from src.utils.api_wrapper import ModelWrapper
        all_api_keys = load_json(api_key_path) 
        api_key = all_api_keys[model_name.lower()]
        api_model = ModelWrapper(model_name=model_name, api_key=api_key)
        api_model.max_tokens = MAX_TOKENS
        api_model.temperature = temp
        api_model.top_p = TOP_P
        api_model.top_k = TOP_K

        logger.info('Generating basic responses')
        cleaned_outputs_basic = [
            api_model.generate_response(i) if i != 'SKIPPED' else 'SKIPPED'
            for i in tqdm(text_basic_list) 
        ]
        
        logger.info('Generating instructive responses')
        cleaned_outputs_instructive = [
            api_model.generate_response(i) if i != 'SKIPPED' else 'SKIPPED' 
            for i in tqdm(text_instructive_list)
        ]

        logger.info('Generating cot responses')
        cleaned_outputs_cot = [
            api_model.generate_response(i) if i != 'SKIPPED' else 'SKIPPED'
            for i in tqdm(text_cot_list)
        ]

        # cleaned_outputs_basic = run_api(model_name=model_name, prompt_format='basic', temp=temp, api_key_path=api_key_path, data_path=data_path, subset=subset)
        # cleaned_outputs_instructive = run_api(model_name=model_name, prompt_format='instructive', temp=temp, api_key_path=api_key_path, data_path=data_path, subset=subset)
        # cleaned_outputs_cot = run_api(model_name=model_name, prompt_format='cot', temp=temp, api_key_path=api_key_path, data_path=data_path, subset=subset)

    else:
        from vllm import LLM, SamplingParams
        import torch
        # hf_token = load_json(api_key_path).get('hf', '')
        # os.environ['HF_TOKEN'] = hf_token
        # Load model using 4 GPUs
        llm = LLM(
            # model=f'{cache_dir}{model_name}', 
            model=f'{model_name}',
            tensor_parallel_size=int(torch.cuda.device_count()),        
            gpu_memory_utilization=0.95, 
            max_model_len=MAX_TOKENS
        )

        # Run inference
        logger.info(f'Running {model_name} with temperature={temp}')

        # Use hyperparameters from original paper (temperature is variable for additional experiments)
        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS, 
            temperature=temp, 
            top_p=TOP_P, 
            top_k=TOP_K
        )

        logger.info('Generating basic prompts')
        skipped_indices_basic = [i for i in range(len(text_basic_list)) if text_basic_list[i] == 'SKIPPED']
        outputs_basic = llm.generate([text for text in text_basic_list if text != 'SKIPPED'], sampling_params)
        cleaned_outputs_basic = [''] * len(text_basic_list)
        output_counter = 0
        for i in range(len(text_basic_list)):
            if i in skipped_indices_basic:
                cleaned_outputs_basic[i] = 'SKIPPED'
            else:
                cleaned_outputs_basic[i] = outputs_basic[output_counter].outputs[0].text
                output_counter += 1

        logger.info('Generating instructive prompts')
        skipped_indices_instructive = [i for i in range(len(text_instructive_list)) if text_instructive_list[i] == 'SKIPPED']
        outputs_instructive = llm.generate([text for text in text_instructive_list if text != 'SKIPPED'], sampling_params)
        cleaned_outputs_instructive = [''] * len(text_instructive_list)
        output_counter = 0
        for i in range(len(text_instructive_list)):
            if i in skipped_indices_instructive:
                cleaned_outputs_instructive[i] = 'SKIPPED'
            else:
                cleaned_outputs_instructive[i] = outputs_instructive[output_counter].outputs[0].text
                output_counter += 1

        logger.info('Generating cot prompts')
        skipped_indices_cot = [i for i in range(len(text_cot_list)) if text_cot_list[i] == 'SKIPPED']
        outputs_cot = llm.generate([text for text in text_cot_list if text != 'SKIPPED'], sampling_params)
        cleaned_outputs_cot = [''] * len(text_cot_list)
        output_counter = 0
        for i in range(len(text_cot_list)):
            if i in skipped_indices_cot:
                cleaned_outputs_cot[i] = 'SKIPPED'
            else:
                cleaned_outputs_cot[i] = outputs_cot[output_counter].outputs[0].text
                output_counter += 1

    for i, item in enumerate(full_data):
        item["output"] = {
            "text_basic": cleaned_outputs_basic[i],
            "text_instructive": cleaned_outputs_instructive[i],
            "text_cot": cleaned_outputs_cot[i]
        }

    # Save the updated data to a JSON file
    model_name_short = '_'.join(model_name.split('/')[1:] if '/' in model_name else [model_name])
    if run_id:
        output_file_name = f'data/outputs/{run_id}/{model_name_short}.json'
    else:
        output_file_name = f'data/outputs/temp_{temp}/{model_name_short}.json'
    save_json(full_data, output_file_name)
    logger.info(f"Inference completed using {model_name}.")
    print(f"Inference completed using {model_name}\n\n\n\n\n.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Conduct inference on creative questions (i.e., Ask LLMs to answer creative questions).')

    parser.add_argument('-model_name', 
                        type=str, 
                        required=True,
                        # default='Qwen2.5-72B-Instruct', 
                        help=f'Name of LLM used for inference (should match name of local directory where weights are stored)'
                        )
    
    parser.add_argument('-temp', 
                        type=int, 
                        default=1, 
                        help=f'Original temp=1; however, it is variable here for additional experiments'
                        )
    
    parser.add_argument('-cache_dir', 
                        type=str, 
                        default='/playpen-ssd/pretrained_models/', 
                        help=f'Path to local directory containing model weights'
                        )
    parser.add_argument('-subset', 
                        type=list, 
                        default=DEFAULT_SUBSET, 
                        help=f'List of subsets to include in the evaluation'
                        )
    parser.add_argument('-api_key_path',
                        type=str,
                        default=os.environ.get(
                            "CREATIVITYPRISM_API_KEYS",
                            "/ihome/xli/joh227/developer/creativityprism_git_workspace/api_keys.json"
                        ),
                        help='Path to API key json file (overrides CREATIVITYPRISM_API_KEYS env var).'
                        )
    parser.add_argument('-data_path',
                        type=str,
                        default='data/processed/ttct.json',
                        help=f'Path to data json file.'
                        )
    parser.add_argument('-prompt_formats',
                        type=str,
                        default='all',
                        help=f'List of settings to include in the evaluation.'
                        )
    parser.add_argument('-run_id',
                        type=str,
                        default="",
                        help=f'Optional run id; when set, outputs go to data/outputs/<run_id>/<model>.json (mirrors -run_id in ttct_evaluation.py).'
                        )
    parser.add_argument('-num_samples',
                        type=int,
                        default=-1,
                        help='Limit inference to first N samples (smoke-test mode). -1 means no limit.'
                        )
    args = parser.parse_args()
    main(**vars(args)) 