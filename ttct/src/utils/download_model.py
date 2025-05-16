"""Download open-source models to local directory
"""
import argparse
import logging
from huggingface_hub import snapshot_download

MODEL_REPOS = {
    'Mistral-7B-Instruct-v0.3'  :'mistralai/Mistral-7B-Instruct-v0.3', 
    'Qwen2.5-7B-Instruct'       :'Qwen/Qwen2.5-7B-Instruct',
    'OLMo-2-1124-7B-Instruct'   :'allenai/OLMo-2-1124-7B-Instruct',
    'Llama-3.1-8B-Instruct'     :'meta-llama/Llama-3.1-8B-Instruct',   
    'OLMo-2-1124-13B-Instruct'  :'allenai/OLMo-2-1124-13B-Instruct',
    'OLMo-2-1124-13B-SFT'       :'allenai/OLMo-2-1124-13B-SFT',   
    'OLMo-2-1124-13B-DPO'       :'allenai/OLMo-2-1124-13B-DPO',
    'Mistral-Small-24B-Instruct-2501':'mistralai/Mistral-Small-24B-Instruct-2501', 
    'Qwen2.5-32B-Instruct'      :'Qwen/Qwen2.5-32B-Instruct',
    'Mixtral-8x7B-Instruct-v0.1':'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'Llama-3.3-70B-Instruct'    :'meta-llama/Llama-3.3-70B-Instruct',
    'Qwen2.5-72B-Instruct'      :'Qwen/Qwen2.5-72B-Instruct'
    }


def main(model_name, cache_dir):
    logger = logging.getLogger(__name__)

    snapshot_download(
        repo_id=MODEL_REPOS[model_name],
        local_dir=f'{cache_dir}{model_name}',
    )

    logger.info(f"Download completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Conduct evaluation on creative outputs from LLMs (using LLM-as-a-judge).')

    parser.add_argument('-model_name', 
                        type=str, 
                        default='Qwen2.5-72B-Instruct', 
                        help=f'Name of LLM being downloaded'
                        )
    
    parser.add_argument('-cache_dir', 
                        type=str, 
                        default='/playpen-ssd/pretrained_models', 
                        help=f'Path to local directory containing model weights'
                        )

    args = parser.parse_args()
    main(**vars(args)) 