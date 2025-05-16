import json
from utils.api_wrapper import ModelWrapper
from tqdm import tqdm
import logging

def get_api_key(model_name):
    '''Set up API keys'''

    if 'gpt' in model_name.lower():
        api_key = ''    # TODO
    if 'claude' in model_name.lower():
        api_key = ''    # TODO
    if 'gemini' in model_name.lower():
        api_key = ''    # TODO
    if 'deepseek' in model_name.lower():
        api_key = ''    # TODO

def run_api(model_name, prompt, temp):
    logger = logging.getLogger(__name__)
    logger.info(f'Getting inference from {model_name} for {prompt} using temperature={temp}')

    api_key = get_api_key(model_name=model_name)

    # Get inputs data
    with open('/playpen-ssd/abrei/creativity_benchmark/ttct/data/processed/ttct.json', "r", encoding="utf-8") as file:
            data = json.load(file)

    inputs = [item['input'][f'text_{prompt}'] for item in data]
    model = ModelWrapper(model_name=model_name, api_key=api_key)

    outputs = []
    for i in tqdm(inputs):
        out = model.generate_response(i)
        outputs.append(out) 
    
    return outputs