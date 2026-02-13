from abc import ABC, abstractmethod
import os 
from api_warpper import ModelWrapper
# Disable parallelism before loading vLLM
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
import torch

class InferenceDriver(ABC):    
    open_source_models = {
        # open-samll
        "mistral-7b-instruct": {
            "hf_dir": "mistralai/Mistral-7B-Instruct-v0.3"
        },       
        "qwen-7b-instruct": {
            "hf_dir": "Qwen/Qwen2.5-7B-Instruct"
        },
        "qwen-coder-7B-instruct": {
            "hf_dir": "Qwen/Qwen2.5-Coder-7B-Instruct"
        },
        "OLMo-7B-instruct": {
            "hf_dir": "allenai/OLMo-2-1124-7B-Instruct"
        },
        "Llama-31-8B-instruct": {
            "hf_dir": "meta-llama/Llama-3.1-8B-Instruct"
        },
        "OLMo2-13B-instruct": {
            "hf_dir": "allenai/OLMo-2-1124-13B-Instruct"
        },
        # open-medium
        "Mistral-24B-instruct": {
            "hf_dir": "mistralai/Mistral-Small-24B-Instruct-2501"
        },
        "qwen-32b-instruct": {
            "hf_dir": "Qwen/Qwen2.5-32B-Instruct"
        },
        "qwen-coder-32b-instruct": {
            "hf_dir": "Qwen/Qwen2.5-Coder-32B-Instruct"
        },
        # open-large
        "Mistral-8x7B-instruct": {
            "hf_dir": "mistralai/Mixtral-8x7B-Instruct-v0.1"
        },
        "Llama-33-70B-instruct": {
            "hf_dir": "meta-llama/Llama-3.3-70B-Instruct"
        },
        "Llama-31-70B-instruct": {
            "hf_dir": "meta-llama/Llama-3.1-70B-Instruct"
        },
        "qwen-72B-instruct": {
            "hf_dir": "Qwen/Qwen2.5-72B-Instruct"
        },
        "OLMo2-13B-dpo": {
            "hf_dir": "allenai/OLMo-2-1124-13B-DPO"
        },
        "OLMo2-13B-sft": {
            "hf_dir": "allenai/OLMo-2-1124-13B-SFT"
        }
    }
    closed_source_model = {
        "gemini-2.0-flash": {
            "token": "key"
        },
        "gpt-4.1-mini": {
            "token": "key"
        },
        "gpt-4.1": {
            "token": "key"
        },
        "deepseek-chat": {
            "token": "key"
        },
        "deepseek-reasoner": {
            "token": "key"
        },
        "claude-3-5-haiku-20241022": {
            "token": "key"
        },
        "claude-3-7-sonnet-20250219": {
            "token": "key"
        }
    }
    def __init__(self, config = {}):
        super().__init__()
        for key in config:
            setattr(self, key, config[key])
        self.config = config 

        if self.model_name in self.open_source_models:
            hf_dir = self.open_source_models[self.model_name]['hf_dir']
            # max_model_len = self.open_source_models[self.model_name].get('max_model_len', None)
            self.llm = LLM(
                model=hf_dir, 
                tensor_parallel_size=torch.cuda.device_count(), 
                dtype="bfloat16",
                max_model_len=4096
            )
            model_config = config.get("model_config", {})
            self.sampling_params = SamplingParams(
                temperature = model_config.get("temperature", 0),
                top_p = model_config.get("top_p", 0.9),
                max_tokens = model_config.get("max_new_tokens", 2000),
                # min_tokens= 20, 
                seed = 42 # model_config.get("seed", 42),
            )
            # print("max_tokens: ", model_config.get("max_new_tokens", 288))
        elif self.model_name in self.closed_source_model:
            access_token = self.closed_source_model[self.model_name]['token']
            self.api_model = ModelWrapper(model_name=self.model_name, api_key=access_token)
        else:
            raise NotImplementedError
        print('=> Done initialization!')

    @abstractmethod
    def create_batched_prompt(self):
        # depends on the subclass
        pass 
    
    @abstractmethod
    def parse_vllm_outputs(self, vllm_results):
        # depends on the subclass
        pass 

    @abstractmethod
    def inference(self):
        # depends on the subclass
        pass 
