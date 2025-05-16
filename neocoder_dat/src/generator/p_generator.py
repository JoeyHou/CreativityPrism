"""
Generator classes for normal prompting and inference
"""
from tqdm import tqdm
import json
from typing import Text, Dict, Any, Union
import logging
import os

from vllm import SamplingParams

from ..models.model import OpenAIModel, AnthropicModel, OpenModel, GenAIModel, DeepSeekModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class APIModelSingleInference:
    
    def __init__(self,
                 model: Union[OpenAIModel, AnthropicModel]
                 ):
        super().__init__()
        self.model = model
    
    def inference(self, 
                  problem_statement: str, 
                  save_path: str,
                  overwrite: bool = False,
                  repeat: int = 1):

        if os.path.exists(save_path) and not overwrite:
            logger.warning(f"File {save_path} exists. To overwrite, please use --overwrite flag.")
            return
        
        records = []

        if type(self.model) == OpenAIModel:
            for n in range(repeat):
                self.model.restart()
                output = self.model(problem_statement)[0]
                records.append({'problem_statement': problem_statement,
                                'output': output})

        elif type(self.model) == AnthropicModel or type(self.model) == GenAIModel or type(self.model) == DeepSeekModel:
            for n in range(repeat):
                if type(self.model) == DeepSeekModel:
                    self.model.restart()
                output = self.model(problem_statement)
                records.append({'problem_statement': problem_statement,
                                'output': output})
        else:
            raise ValueError(f'Unsupported model type')

        with open(save_path, "w") as f:
            json.dump(records, f, indent=4)

        if hasattr(self.model, 'gpt_usage'):
            logger.warning(f'Usage: {self.model.gpt_usage()}')

class OpenModelSingleInference:
    
    def __init__(self,
                 model: OpenModel,
                 config: Dict[Text, Any],
                 use_vllm: bool = False):
    
        super().__init__()
    
        self.model = model
        self.config = config
        self.use_vllm = use_vllm
    
    def inference(self, 
                  problem_statement: str, 
                  save_path: str,
                  overwrite: bool = False,
                  repeat: int = 1):
        
        if os.path.exists(save_path) and not overwrite:
            logger.warning(f"File {save_path} exists. To overwrite, please use --overwrite flag.")
            return

        if self.use_vllm:
            logger.info(f'Using VLLM for inference')
            config = SamplingParams(**self.config)
            records = self.vllm_inference(problem_statement, config, repeat)
        else:
            logger.info(f'Using HF for inference')
            records = self.hf_inference(problem_statement, self.config, repeat)

        with open(save_path, "w") as f:
            json.dump(records, f, indent=4)
    
    def vllm_inference(self,
                       problem_statement: str, 
                       config: SamplingParams,
                       repeat: int):
        if "inst" in self.model.model_name.lower():
            problem_statement = self.model.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": problem_statement}
                ],
                tokenize=False
            )
        records = []
        for n in tqdm(range(repeat), desc="Inferencing"):

            results = self.model.model.generate([problem_statement], 
                                                sampling_params=config)
            outputs = [result.outputs[0].text for result in results]

            records.append({'problem_statement': problem_statement,
                            'outputs': outputs})
        return records
        
    def hf_inference(self, 
                     problem_statement: str,
                     config: Dict[Text, Any],
                     repeat: int):

        raise NotImplementedError("HF inference is not implemented yet.")