import json
from utils.api_wrapper import ModelWrapper
from src.utils.helpers import load_json
from tqdm import tqdm
import logging

def get_api_key(model_name, api_key_path):
    '''Set up API keys'''
    api_keys = load_json(api_key_path) 
    return api_keys[model_name.lower()]

def run_api(model_name, prompt, temp, api_key_path, data_path = 'data/processed/ttct.json'):
    logger = logging.getLogger(__name__)
    logger.info(f'Getting inference from {model_name} for {prompt} using temperature={temp}')

    api_key = get_api_key(model_name=model_name, api_key_path=api_key_path)

    # Get inputs data
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    inputs = [item['input'][f'text_{prompt}'] for item in data]
    model = ModelWrapper(model_name=model_name, api_key=api_key)

    outputs = []
    for i in tqdm(inputs):
        out = model.generate_response(i)
        outputs.append(out) 
    
    return outputs