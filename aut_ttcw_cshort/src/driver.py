from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
import torch 

from src.utils.api_wrapper import ModelWrapper
from src.utils.helpers import load_json

class Driver(ABC):
    
    def __init__(self, config = {}):
        super().__init__()

        # 0. basic setups
        self.config = config 
        for key in config:
            setattr(self, key, config[key])
        self.logger = config['logger']

        # 1. initialze class info
        self.open_source_models = {
            "mistral_7b_instruct": {
                "hf_dir": "mistralai/Mistral-7B-Instruct-v0.3"
            },
            "llama3_8b_instruct": {
                "hf_dir": "meta-llama/Llama-3.1-8B-Instruct",
            },
            "llama3_70b_instruct": {
                "hf_dir": "meta-llama/Llama-3.3-70B-Instruct",
                "max_model_len": 8000
            },
            "qwen_7b_instruct": {
                "hf_dir": "Qwen/Qwen2.5-7B-Instruct"
            },
            "qwen_32b_instruct": {
                "hf_dir": "Qwen/Qwen2.5-32B-Instruct",
                "max_model_len": 8000
            },
            "qwen_72b_instruct": {
                "hf_dir": "Qwen/Qwen2.5-72B-Instruct",
                "max_model_len": 8000
            },
            "qwen_72b_instruct_long": {
                "hf_dir": "Qwen/Qwen2.5-72B-Instruct",
                "max_model_len": 12000
            },
            "deepseek_llama_70b": {
                "hf_dir": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "max_model_len": 8000
            },
            "deepseek_qwen_32b": {
                "hf_dir": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "max_model_len": 8000
            },
            "deepseek_qwen_7b": {
                "hf_dir": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            },
            "olmo_7b": {
                "hf_dir": "allenai/OLMo-2-1124-7B-Instruct"
            },
            "olmo_13b": {
                "hf_dir": "allenai/OLMo-2-1124-13B-Instruct"
            },
            "mistral_small_24b": {
                "hf_dir": "mistralai/Mistral-Small-24B-Instruct-2501"
            },
            "mixtral_8x7b": {
                "hf_dir": "mistralai/Mixtral-8x7B-Instruct-v0.1"
            },
            "olmo_13b_dpo": {
                "hf_dir": "allenai/OLMo-2-1124-13B-DPO"
            },
            "olmo_13b_sft": {
                "hf_dir": "allenai/OLMo-2-1124-13B-SFT"
            },
        }
        self.api_models = load_json('./api_keys.json')  ### Change if you are not Joey
        self.parsing_output_len = 256 # length limit of parsing output 
        self.local_parsing_model = 'Qwen/Qwen2.5-7B-Instruct'
        self.test_size = config.get('test_size', 10e5) # test_size is large by default
        
        # 2. initialize llm
        if self.model_name in self.open_source_models:
            self.use_open_model = True
            self.api_key = ''
            hf_dir = self.open_source_models[self.model_name]['hf_dir']
            max_model_len = self.open_source_models[self.model_name].get('max_model_len', None)
            self.llm = LLM(
                model=hf_dir, 
                tensor_parallel_size=torch.cuda.device_count(), 
                dtype="bfloat16", 
                max_model_len=max_model_len
            )
            self.sampling_params = SamplingParams(
                temperature = config.get("temperature", 0.75), 
                top_p = config.get("top_p", 1),
                max_tokens = config.get("max_tokens", 4096)
            )
        elif self.model_name in self.api_models:
            self.use_open_model = False
            self.api_key = self.api_models[self.model_name]
            self.llm = ModelWrapper(self.model_name, self.api_key)
            self.sampling_params = None 
        else:
            raise NotImplementedError
        
        # 3. ending
        self.logger.debug('config:' + str(config))
        self.logger.debug('=> Done initialization!')
