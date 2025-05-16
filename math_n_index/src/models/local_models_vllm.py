# models/local_models.py
import logging
import json
import torch
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# from vllm.utils import get_openai_chat_template
def load_config(input_file):
    file_path = input_file
    with open(file_path, "r") as file:
        return json.load(file)

from src.utils import extract_yes_no, load_json, save_json, setup_logger

config = load_config(input_file="configs/creative_math_config.json")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The dictionary maps each local model_name to a vLLM LLM instance
_LOCAL_VLLM_MODELS = {}
_LOCAL_VLLM_TOKENIZERS = {}  # New cache for tokenizers

max_new_tokens = config["model_config"]["max_new_tokens"]
# do_sample = config["model_config"]["do_sample"]
temperature = config["model_config"].get("temperature", 0.0)  # Get value or default to 0.0

# Ensure temperature is a valid float, otherwise fall back to 0.0
if not isinstance(temperature, (int, float)):
    temperature = 0.0

top_k = config["model_config"]["top_k"]
top_p = config["model_config"]["top_p"]
TENSOR_PARRALLEL_SIZE = config["model_config"]["tensor_parrallel_size"]

def load_local_model_vllm(model_name):
    """
    Loads (or retrieves from cache) a vLLM model for `model_name`.
    Returns (model, tokenizer), where `model` is the vLLM LLM object,
    and `tokenizer` is AutoTokenizer (if needed).
    """
    logger = logging.getLogger(__name__)
    
    # Get model ID from config
    model_id = config["model_version"].get(model_name)
    if not model_id:
        raise ValueError(f"Local model {model_name} not found in config['model_version'].")

    # Load the model if it's not already cached
    if model_name not in _LOCAL_VLLM_MODELS:
        logger.info(f"[vLLM] Loading local model '{model_name}' from '{model_id}' ...")
        # for now just use all the available GPUs
        # use bfloat 16 for consistency
        _LOCAL_VLLM_MODELS[model_name] = LLM(model_id, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16")
    else:
        logger.info(f"[vLLM] Reusing loaded model '{model_name}'.")

    # Load the tokenizer if it's not already cached
    if model_name not in _LOCAL_VLLM_TOKENIZERS:
        logger.info(f"[vLLM] Loading tokenizer for '{model_name}' ...")
        _LOCAL_VLLM_TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        logger.info(f"[vLLM] Reusing tokenizer for '{model_name}'.")

    # Return (model, tokenizer) to match original function signature
    return _LOCAL_VLLM_MODELS[model_name], _LOCAL_VLLM_TOKENIZERS[model_name]


def generate_local_response_vllm(model_name, model, tokenizer, messages):
    """
    Generates a response using vLLM.
    - `model` is a vLLM LLM instance.
    - `messages` is a list of structured message dicts.
    """
    logger = logging.getLogger(__name__)
    
    # Ensure messages is in the correct format (list of dictionaries)
    if isinstance(messages, str):
        logger.info("Converting the message to expected format!")
        messages = [{"role": "user", "content": messages}]
    
    # Convert structured messages into a text prompt.
    # Use tokenize=False because vLLM.generate expects a raw text string.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False # we only need raw prompts
    )

    # TODO: decide what parameters are needed
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # vLLM expects a list of prompt strings.
    outputs = model.generate([prompt_text], sampling_params)

    # Ensure output exists before extracting
    if outputs and outputs[0].outputs:
        response_text = outputs[0].outputs[0].text.strip()
    else:
        response_text = ""  # Fallback to empty response if nothing was generated
    logger.info(f"[vLLM] Generated response for model '{model_name}': {response_text[:50]}...")
    return response_text


def generate_local_batch_response_vllm(model_name, model, tokenizer, messages_list):
    """
    Generates responses for a batch of prompts using vLLM.
    
    Args:
        model_name (str): The name of the model.
        model: A vLLM model instance.
        tokenizer: The tokenizer associated with the model.
        messages_list (list): A list where each element is either a string or a list of structured message dictionaries.
        
    Returns:
        list: A list of response strings corresponding to each prompt.
    """
    logger = logging.getLogger(__name__)
    
    # Convert each input into a properly formatted prompt string.
    prompt_texts = []
    for messages in messages_list:
        # Ensure messages is in the expected list-of-dicts format.
        if isinstance(messages, str):
            logger.info("Converting the message to expected format!")
            messages = [{"role": "user", "content": messages}]
        
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # We only need raw prompt text.
        )
        prompt_texts.append(prompt_text)
    
    # Define sampling parameters. These variables (max_new_tokens, temperature, top_p, top_k)
    # should be defined globally.
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    
    # vLLM expects a list of prompt strings.
    outputs = model.generate(prompt_texts, sampling_params)
    
    responses = []
    for output in outputs:
        # Check if there are generated outputs; if so, extract the first generation.
        if output.outputs:
            response_text = output.outputs[0].text.strip()
        else:
            response_text = ""  # Fallback to empty response if nothing was generated.
        responses.append(response_text)
        logger.info(f"[vLLM] Generated response for model '{model_name}': {response_text[:50]}...")
    
    return responses
