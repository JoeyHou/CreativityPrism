import json
import logging
import os
 
def load_json(data_path):
    logger = logging.getLogger(__name__)
    with open(data_path, "r") as file:
        data = json.load(file)
    logger.info(f"JSON data loaded from {os.path.abspath(data_path)}")
    return data

def save_json(data, data_path):
    logger = logging.getLogger(__name__)

    # Ensure the directory exists
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory {directory} created.")
        
    with open(data_path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(data_path)}")


def load_prompts_as_list(data_path, prompt_type):
    logger = logging.getLogger(__name__)
    with open(data_path, "r") as file:
        data = json.load(file)
    logger.info(f"JSON data loaded from {os.path.abspath(data_path)}")
    logger.info(f"Retrieving {prompt_type} prompts")
    prompt_list = [item["input"][prompt_type] for item in data]
    return prompt_list

def load_evaluations_as_list(data_path, prompt_type):
    logger = logging.getLogger(__name__)
    with open(data_path, "r") as file:
        data = json.load(file)
    logger.info(f"JSON data loaded from {os.path.abspath(data_path)}")
    logger.info(f"Retrieving {prompt_type} prompts")
    eval_list = [item["evaluation"][prompt_type] for item in data]
    return eval_list
