from configs import config
# from src.models.api_models import generate_api_response, load_api_model
from src.models.local_models import generate_local_response, load_local_model
from src.models.local_models_vllm import generate_local_response_vllm, load_local_model_vllm, generate_local_batch_response_vllm
from src.models.prompt_utils import load_messages, load_messages_vllm

CONFIG = config["model_config"]


class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_api_model = model_name in [
            "claude-3-opus",
            "claude-3-5-sonnet",
            "deepseek-v2",
            "gemini-1.5-pro",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        if self.is_api_model:
            # self.model = load_api_model(model_name)
            return "No API Models"
        else:
            self.model, self.tokenizer = load_local_model(model_name)

    def generate_response(self, prompt):
        messages = load_messages(self.model_name, prompt)
        if self.is_api_model:
            # return generate_api_response(self.model_name, self.model, messages)
            return "None generation"
        else:
            return generate_local_response(
                self.model_name, self.model, self.tokenizer, messages
            )

# models/__init__.py or models/api_models.py or ...
# references:
#   from models.local_models import load_local_model, generate_local_response
#   from models.api_models import load_api_model, generate_api_response
# TODO: please check. this should be the same as the one above but just added for not confusing them together
class ModelWrapperVLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_api_model = model_name in [
            "claude-3-opus",
            "claude-3-5-sonnet",
            "deepseek-v2",
            "gemini-1.5-pro",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        if self.is_api_model: # not revised now
            # self.model = load_api_model(model_name)
            return "No API Models"
        else:
            self.model, self.tokenizer = load_local_model_vllm(model_name)

    def generate_response(self, prompt):
        prompt_text = load_messages_vllm(self.model_name, prompt)  # Already formatted!
    
        if self.is_api_model:
            # return generate_api_response(self.model_name, self.model, prompt_text)
            return "No API Models"
        else:
            return generate_local_response_vllm(self.model_name, self.model, self.tokenizer, prompt_text)
    
    def generate_batch_response(self, prompts):
        """
        Performs batched inference on a list of prompts using vLLM.
        
        Args:
            prompts (list): A list of prompt inputs. Each can be a raw string or structured messages.
        
        Returns:
            list: A list of responses corresponding to each prompt.
        """
        if self.is_api_model:
            # Implement API batch inference if needed.
            return ["No API Models" for _ in prompts]
        else:
            # Convert each prompt using load_messages_vllm to ensure proper formatting.
            formatted_prompts = [load_messages_vllm(self.model_name, prompt) for prompt in prompts]
            return generate_local_batch_response_vllm(
                self.model_name, self.model, self.tokenizer, formatted_prompts
            )
