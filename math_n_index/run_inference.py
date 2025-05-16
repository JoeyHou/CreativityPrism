import sys
import argparse
import torch
import gc
import os

from src.utils.helpers import load_json
from src.inference.creative_writing import CreativeWritingInference
from src.inference.creative_math_inference import CreativeMathInference


def run_inference(config):

    for exp_config in config['experiments_list'][0:1]: # next is 3.3
        task = exp_config.get('task', 'creative_writing')
        if task == 'creative_writing':
            inference_driver = CreativeWritingInference(exp_config)
        elif task == 'creative_math':
            inference_driver = CreativeMathInference(exp_config)
        elif task == 'creative_index':
            inference_driver = None
        else:
            raise NotImplementedError
        
        try:
            print(f"Starting inference for {exp_config['model_name']}...")
            inference_driver.inference()
            print(f"Inference completed for {exp_config['model_name']}.")

        except Exception as e:
            print(f"Error during inference for {exp_config['model_name']}: {e}")
            continue  # Move to the next model even if one fails

        finally:
            # **Forcefully delete the model & clear memory**
            print(f"Releasing memory for {exp_config['model_name']}...")
            del inference_driver  # Remove the model instance
            torch.cuda.empty_cache()  # Free GPU memory
            gc.collect()  # Force Python garbage collection
            
            # Kill any remaining vLLM processes to avoid conflicts
            print("Killing any remaining vLLM processes...")
            os.system("pkill -f 'python.*vllm'")
        


if __name__ == '__main__':
    if len(sys.argv) == 1:
        config = load_json('configs/default.json') # default config
    else:
        config = load_json(sys.argv[1]) # load configs/creative_math_config.json and others
    # print(config)
    run_inference(config)