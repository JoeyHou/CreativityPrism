"""Open- and closed- source model for generating rationales
"""
import os
import json
import random
import backoff
import openai
import transformers
import torch
import boto3
import anthropic
import numpy as np

from abc import ABC, abstractmethod
from overrides import overrides
from openai import OpenAI
from google import genai
from google.genai import types
from vllm import LLM

CACHE_DIR="/scratch365/ylu33/hf-models"

completion_tokens = prompt_tokens = 0

prompt_cache_miss_tokens = prompt_cache_hit_tokens = 0

class ds_args(object):
    """Dummy argument class.
    The object will be passed to deepspeed.initialize()
    if needed for data parallelism.
    """
    def __init__(self,
                 local_rank: int,
                 deepspeed_config: str,
                 seed: int,
                 deepspeed: bool
                 ):
        super().__init__()
        self.local_rank = local_rank
        self.deepspeed_config = deepspeed_config
        self.deepspeed = deepspeed
        self.seed = seed

class OpenModel(ABC):
    """
    """
    def __init__(self,
                 model_name: str,
                 prompt: str):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt

    @abstractmethod
    def load_model(self):
        """
        """
        raise NotImplementedError("Model is an abstract class.")

class OpenModelVLLM(OpenModel):
    def __init__(self,
                 model_name: str,
                 prompt: str):
        
        super().__init__(model_name, prompt)
        self.load_model()

    @overrides
    def load_model(self):
        num_gpus = torch.cuda.device_count()
        self.model = LLM(model=self.model_name, 
                         download_dir=CACHE_DIR, 
                         tensor_parallel_size=num_gpus)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

class OpenModelHF(OpenModel):
    def __init__(self,
                 model_name: str,
                 prompt: str):
        
        super().__init__(model_name, prompt)    
        self.load_model()

    @overrides
    def load_model(self):
        args = ds_args(local_rank=0, 
                       deepspeed_config="src/utils/ds_config.json", 
                       deepspeed=True,
                       seed=42)
        self.initialize(args)
        if "t5" in self.model_name.lower():
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_name, 
                                                                            cache_dir=CACHE_DIR,
                                                                            trust_remote_code=True,
                                                                            device_map="auto",
                                                                            torch_dtype=torch.float16
                                                                            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                                           cache_dir=CACHE_DIR,
                                                                           trust_remote_code=True,
                                                                           device_map="auto",
                                                                           torch_dtype=torch.float16
                                                                           )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def initialize(self, args):

        self.init_distributed(args)

        self.set_random_seed(args.seed)

    def set_random_seed(self, seed):
        """Set random seed for reproducability.
        """
        # seed = dist.get_rank() + seed
        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def init_distributed(self, args):
        """Initialize distributed inference.
        """
        args.rank = int(os.environ["SLURM_PROCID"]) # this is the rank of the current GPU
        args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node) # this is the rank of the current GPU within the node
        args.world_size = int(os.getenv("WORLD_SIZE", "1")) # this is the number of GPUs

        if args.rank == 0:
            print(f"using world size: {args.world_size}")

        # Manually set the device ids.
        self.device = args.local_rank
        torch.cuda.set_device(self.device)

        # dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size, timeout=timedelta(seconds=30))

class OpenAIModel:
    def __init__(self, 
                 model,
                 temperature,
                 top_p,
                 max_tokens,
                 n: int = 1,
                 gpt_setting: str = None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n
        self.gpt_setting = gpt_setting
        self.restart()
        self.client = OpenAI()
        
    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def chatcompletions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def completions_with_backoff(self, **kwargs):
        return self.client.completions.create(**kwargs)

    def chatgpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.chatcompletions_with_backoff(model=self.model, 
                                                messages=self.message, 
                                                temperature=self.temperature, 
                                                top_p=self.top_p,
                                                max_tokens=self.max_tokens, 
                                                n=self.n)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        return outputs

    def completiongpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.completions_with_backoff(model=self.model, 
                                            messages=self.message, 
                                            temperature=self.temperature, 
                                            top_p=self.top_p,
                                            max_tokens=self.max_tokens, 
                                            n=self.n)
        outputs.extend([choice.text for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        return outputs

    @staticmethod
    def gpt_usage(model="gpt-4-turbo"):
        global completion_tokens, prompt_tokens
        if model == "gpt-4": # Currently points to gpt-4-0613
            cost = completion_tokens / 1000000 * 60 + prompt_tokens / 1000000 * 30
        elif model == "gpt-4.1": # Currently points to gpt-4.1-2025-04-14
            cost = completion_tokens / 1000000 * 8 + prompt_tokens / 1000000 * 2
        elif model == "gpt-4.1-mini": # Currently points to gpt-4.1-mini-2025-04-14
            cost = completion_tokens / 1000000 * 1.6 + prompt_tokens / 1000000 * 0.4
        elif model == "gpt-4-turbo": # The latest GPT-4 Turbo model with vision capabilities
            cost = completion_tokens / 1000000 * 30 + prompt_tokens / 1000000 * 10
        elif model == "gpt-3.5-turbo": # Currently points to gpt-3.5-turbo-0125
            cost = completion_tokens / 1000000 * 1.5 + prompt_tokens / 1000000 * 0.5
        elif model == "gpt-3.5-turbo-instruct":
            cost = completion_tokens / 1000000 * 2 + prompt_tokens / 1000000 * 1.5
        elif "davinci" in model:
            cost = completion_tokens / 1000000 * 2 + prompt_tokens / 1000000 * 2
        else:
            raise ValueError(f"Unknown model: {model}")
        return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


    def __call__(self, input: str) -> list:
        if "davinci" in self.model:
            self.message = self.message + "\nInput: " + input
            return self.completiongpt()
        else:
            self.message.append({"role": "user", "content": input})
            return self.chatgpt()
        
    def update_message(self, output: str):
        if "davinci" in self.model:
            self.message = self.message + "\nOutput: " + output
        else:
            self.message.append({"role": "assistant", "content": output})
    
    def restart(self):
        if "davinci" in self.model:
            self.message = ""
        else:
            self.message = [{"role": "system", "content": self.gpt_setting}] if self.gpt_setting else []

class GenAIModel:

    def __init__(self, 
                 model,
                 temperature,
                 top_p,
                 max_tokens,
                 gpt_setting: str = None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.gpt_setting = gpt_setting

        self.client = genai.Client(
            api_key=os.environ["GENAI_API_KEY"],
        )
    
    @backoff.on_exception(backoff.expo, Exception, max_time=60)
    def __call__(self, input: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.gpt_setting,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                topP=self.top_p
            ),
            contents=[input]
        )

        if response.text == "":
            raise Exception("Empty response from GenAI model")

        return response.text

class AnthropicModel:

    def __init__(self, 
                 model,
                 temperature,
                 top_p,
                 max_tokens,
                 gpt_setting: str = None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.gpt_setting = gpt_setting

        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    
    @backoff.on_exception(backoff.expo, Exception, max_time=60)
    def __call__(self, input: str) -> str:

        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
            system=self.gpt_setting if self.gpt_setting else "NOT_GIVEN",
            messages=[
                {"role": "user", "content": input}
            ]
        )

        if response.content == "":
            raise Exception("Empty response from Anthropic model")

        return response.content[0].text

class DeepSeekModel:
    def __init__(self, 
                 model,
                 temperature,
                 top_p,
                 max_tokens,
                 gpt_setting: str = None
                 ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_setting = gpt_setting

        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

        self.restart()

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=60, max_tries=10)
    def chatcompletions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def get_response(self, prompt: str):
        global completion_tokens, prompt_cache_miss_tokens, prompt_cache_hit_tokens

        self.message.append({"role": "user", "content": prompt})
        
        try:
            completion = self.chatcompletions_with_backoff(
                model=self.model,
                messages=self.message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=1,
                stream=False
            )
            if completion is None:
                print("Failed to get a valid response from DeepSeek API after retries")
                return ""
        except json.decoder.JSONDecodeError:
            # DeepSeek returns an empty response
            return ""

        completion_tokens += completion.usage.completion_tokens
        prompt_cache_miss_tokens += completion.usage.prompt_cache_miss_tokens
        prompt_cache_hit_tokens += completion.usage.prompt_cache_hit_tokens

        return completion.choices[0].message.content
    
    def __call__(self, prompt):
        return self.get_response(prompt)

    def restart(self):
        self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []

    @staticmethod
    def gpt_usage(model="deepseek-chat"):
        global completion_tokens, prompt_cache_miss_tokens, prompt_cache_hit_tokens
        if model == "deepseek-chat": # Currently points to DeepSeek-v3
            cost = completion_tokens / 1000000 * 1.10 + prompt_cache_miss_tokens / 1000000 * 0.27 + prompt_cache_hit_tokens / 1000000 * 0.07
        elif model == "deepseek-reasoner": # Currently points to DeepSeek-r1
            cost = completion_tokens / 1000000 * 2.19 + prompt_cache_miss_tokens / 1000000 * 0.55 + prompt_cache_hit_tokens / 1000000 * 0.14
        else:
            raise ValueError(f"Unknown model: {model}")
        
        return {"completion_tokens": completion_tokens, "prompt_cache_miss_tokens": prompt_cache_miss_tokens, "prompt_cache_hit_tokens": prompt_cache_hit_tokens, "cost": cost}

# class AnthropicModel:

#     def __init__(self, 
#                  model,
#                  temperature,
#                  top_p,
#                  max_tokens,
#                  gpt_setting: str = None):
#         self.model = model
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens
#         self.gpt_setting = gpt_setting
        
#         AWS_KEY = os.environ["AWS_KEY"]
#         AWS_SECRET = os.environ["AWS_SECRET"]

#         if AWS_KEY == None or AWS_SECRET == None:
#             raise Exception("AWS_KEY or AWS_SECRET not found")
        
#         self.bedrock = boto3.client(service_name='bedrock-runtime',
#                                     region_name='us-east-1',
#                                     aws_access_key_id=AWS_KEY,
#                                     aws_secret_access_key=AWS_SECRET)
    
#     @backoff.on_exception(backoff.expo, Exception, max_time=60)
#     def __call__(self, input: str) -> str:
#         body = json.dumps({
#             "anthropic_version": "bedrock-2023-05-31",
#             "max_tokens": self.max_tokens,
#             "temperature": self.temperature,
#             "top_p": self.top_p,
#             "system": self.gpt_setting,
#             "messages": [{"role": "user", "content": input}],
#         })

#         response = self.bedrock.invoke_model(body=body, modelId=self.model)
#         response_body = json.loads(response.get("body").read())
#         result = response_body.get("content")[0]["text"]

#         return result
