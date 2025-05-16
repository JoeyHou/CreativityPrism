import logging
import sys
import time

# revise later
# import google.generativeai as genai
# from anthropic import Anthropic
# from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI

from configs import config

max_new_tokens = config["model_config"]["max_new_tokens"]
temperature = config["model_config"]["temperature"]

'''
def load_api_model(model_name):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)
    api_keys = config["api_keys"]

    if model_name in ["claude-3-opus", "claude-3-5-sonnet"]:
        client = Anthropic(api_key=api_keys["ANTHROPIC_API_KEY"])
    elif model_name == "deepseek-v2":
        logger.warning(
            "The DeepSeek API has recently been upgraded to deepSeek-v3. "
            "To reproduce the results of deepSeek-v2, you may refer to the Hugging Face page at: "
            "https://huggingface.co/deepseek-ai/DeepSeek-V2"
        )
        client = OpenAI(
            api_key=api_keys["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif model_name == "gemini-1.5-pro":
        genai.configure(api_key=api_keys["GEMINI_API_KEY"])
        client = genai.GenerativeModel(model_id)
    elif model_name in ["gpt-4", "gpt-4o", "gpt-4o-mini"]:
        client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])
    else:
        logger.error(f"API model {model_name} is not supported.")
        raise ValueError(f"API model {model_name} is not supported.")

    logger.info(f"Loading model {model_name}...")
    return client


def generate_api_response(model_name, client, messages):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)

    if model_name in ["claude-3-opus", "claude-3-5-sonnet"]:
        message = client.messages.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system="",
            messages=messages,
        )
        response = message.content[0].text
    elif model_name in ["deepseek-v2", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
        completion = client.chat.completions.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=messages,
        )
        response = completion.choices[0].message.content
    elif model_name == "gemini-1.5-pro":
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )
        max_retries = 10  # Maximum number of retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.generate_content(
                    messages, generation_config=generation_config
                ).text
                time.sleep(1)
                break
            except ResourceExhausted as e:
                logger.error(
                    f"Resource exhausted. Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})"
                )
                time.sleep(5)
                retry_count += 1
        if retry_count == max_retries:
            logger.error(f"Max retries reached. Could not complete the request.")
            sys.exit(1)  # Stop the program
    else:
        logger.error(f"Response generation for {model_name} is not implemented.")
        raise ValueError(f"Response generation for {model_name} is not implemented.")

    return response.strip()  # Clean whitespaces
'''