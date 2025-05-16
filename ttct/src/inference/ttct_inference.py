"""Inference for creative questions for seven tasks
"""
import argparse
import logging
from vllm import LLM, SamplingParams
from src.utils.helpers import load_json, load_prompts_as_list, save_json
from src.utils.run_api import run_api


def main(model_name, temp, cache_dir):

    logger = logging.getLogger(__name__)

    if 'gpt' in model_name.lower() or 'claude' in model_name.lower() or 'gemini' in model_name.lower() or 'deepseek' in model_name.lower():
        cleaned_outputs_basic = run_api(model_name=model_name, prompt='basic', temp=temp)
        cleaned_outputs_instructive = run_api(model_name=model_name, prompt='instructive', temp=temp)
        cleaned_outputs_cot = run_api(model_name=model_name, prompt='cot', temp=temp)

    else:
        # Load model using 4 GPUs
        llm = LLM(
                model=f'{cache_dir}{model_name}', 
                tensor_parallel_size=4,        
                gpu_memory_utilization=0.95, 
                max_model_len=2048
                )

        # Load creative questions from original paper
        data_path = 'data/processed/ttct.json'
        data = load_json(data_path)

        # Get data for all 3 prompt types
        text_basic_list = load_prompts_as_list(data_path=data_path, prompt_type="text_basic")
        text_instructive_list = load_prompts_as_list(data_path=data_path, prompt_type="text_instructive")
        text_cot_list = load_prompts_as_list(data_path=data_path, prompt_type="text_cot")

        # Run inference
        logger.info(f'Running {model_name} with temperature={temp}')

        # Use hyperparameters from original paper (temperature is variable for additional experiments)
        sampling_params = SamplingParams(max_tokens=512, 
                                        temperature=temp, 
                                        top_p=1, 
                                        top_k=50)

        logger.info('Generating basic prompts')
        outputs_basic = llm.generate(text_basic_list, sampling_params)
        cleaned_outputs_basic = [output.outputs[0].text for output in outputs_basic]

        logger.info('Generating instructive prompts')
        outputs_instructive = llm.generate(text_instructive_list, sampling_params)
        cleaned_outputs_instructive = [output.outputs[0].text for output in outputs_instructive]

        logger.info('Generating cot prompts')
        outputs_cot = llm.generate(text_cot_list, sampling_params)
        cleaned_outputs_cot = [output.outputs[0].text for output in outputs_cot]

    for i, item in enumerate(data):
        item["output"] = {
            "text_basic": cleaned_outputs_basic[i],
            "text_instructive": cleaned_outputs_instructive[i],
            "text_cot": cleaned_outputs_cot[i]
        }

    # Save the updated data to a JSON file
    output_file_name = f'data/outputs/temp_{temp}/{model_name}.json'
    save_json(data, output_file_name)
    logger.info(f"Inference completed using {model_name}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Conduct inference on creative questions (i.e., Ask LLMs to answer creative questions).')

    parser.add_argument('-model_name', 
                        type=str, 
                        default='Qwen2.5-72B-Instruct', 
                        help=f'Name of LLM used for inference (should match name of local directory where weights are stored)'
                        )
    
    # parser.add_argument('-basic', type=bool, default=True, help=f'Include arg to run basic prompt types')
    # parser.add_argument('-instructive', type=bool, default=True, help=f'Include arg to run instructive prompt types')
    # parser.add_argument('-cot', type=bool, default=True, help=f'Include arg to run chain-of-thought prompt types')
    
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

    args = parser.parse_args()
    main(**vars(args)) 