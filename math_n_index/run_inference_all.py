import sys

from src.utils.helpers import load_json
from src.inference.creative_writing import CreativeWritingInference
from src.inference.creative_math_inference import CreativeMathInference
from src.inference.creative_index_inference import CreativeIndexInference
from src.inference.creative_math_selfimprove import CreativeMathImprove
def run_inference(exp_config):
    task = exp_config.get('task', 'creative_writing')

    if task == 'creative_writing':
        inference_driver = CreativeWritingInference(exp_config)
    elif task == 'creative_math':
        inference_driver = CreativeMathInference(exp_config)
    elif task == 'book' or task == "poem" or task == "speech":
        inference_driver = CreativeIndexInference(exp_config)
    elif task == 'math_improve':
        inference_driver = CreativeMathImprove(exp_config)
    else:
        raise NotImplementedError

    print(f"Starting inference for {exp_config['model_name']}...")
    inference_driver.inference()
    print(f"Inference completed for {exp_config['model_name']}.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run_inference.py <config_file> <experiment_index>")
        sys.exit(1)

    config_file = sys.argv[1]
    experiment_index = int(sys.argv[2])

    config = load_json(config_file)  # Load the full config
    exp_config = config['experiments_list'][experiment_index]  # Select one model config

    run_inference(exp_config)
# creative index: python run_inference_all.py configs/inference_creative_index_api.json 0
# creative math: python run_inference_all.py configs/inference_creative_math_api.json 0
# python run_inference_all.py configs/inference_creative_index.json 5
# python run_inference_all.py configs/inference_creative_index_book.json 0