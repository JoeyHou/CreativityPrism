"""
Classes for denial prompting / constraint-based generation 
"""

import random
from tqdm import tqdm
import json
from typing import Text, List, Dict, Any, Union
from overrides import overrides
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from vllm import SamplingParams

from ..models.model import OpenAIModel, AnthropicModel, OpenModel, GenAIModel, DeepSeekModel
from .generator import CodeGenerator
from ..evaluators.evaluation_utils import enumerate_resume, write_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

class APIModelParallelThreadDPInference(CodeGenerator):
    
    def __init__(self,
                 model: Union[OpenAIModel, AnthropicModel],
                 dp_rounds: int):
        super().__init__()
        
        self.model = model
        self.dp_rounds = dp_rounds
    
    @overrides
    def inference(self, 
                  dataloader: DataLoader, 
                  save_path: str,
                  overwrite: bool = False):
        """Infernce on DP benchmark dataset in parallel threads.
        Each problem is answered in separate openai calls.
        """

        # if os.path.exists(save_path) and not overwrite:
        #     logger.warning(f"File {save_path} exists. To overwrite, please use --overwrite flag.")
        #     return
        
        records = []
        for batch in enumerate_resume(dataloader, save_path):

            problem_statements: List[Text] = batch['inputs'] # 1 x dp_rounds
            problem_id: str = batch['problem_ids'][0]
            codes, outputs = [], []

            if type(self.model) == OpenAIModel:
                for idx, problem_statement in enumerate(problem_statements):
                    self.model.restart()
                    output = self.model(problem_statement)[0]
                    code = self.parse_response(output)
                    if code is None:
                        logger.warning(f'No code generated at {idx+1}th denial iter for {problem_id}')

                    codes.append(code)
                    outputs.append(output)

                record = {'problem_id': problem_id,
                          'problem_statements': problem_statements,
                          'codes': codes,
                          'outputs': outputs,
                          'constraints': batch['constraints'][0]}

                records.append(record)
                
            elif type(self.model) == AnthropicModel or type(self.model) == GenAIModel or type(self.model) == DeepSeekModel:
                for idx, problem_statement in enumerate(problem_statements):
                    if type(self.model) == DeepSeekModel:
                        self.model.restart()

                    output = self.model(problem_statement)
                    outputs.append(output)

                record = {'problem_id': problem_id,
                          'problem_statements': problem_statements,
                          'outputs': outputs,
                          'constraints': batch['constraints'][0]}

                records.append(record)
            else:
                raise ValueError(f'Unsupported model type')
            
            write_jsonl(save_path, [record], append=True)

        # with open(save_path, "w") as f:
        #     json.dump(records, f, indent=4)


        if hasattr(self.model, 'gpt_usage'):
            logger.warning(f'Usage: {self.model.gpt_usage()}')
            
class OpenModelParallelThreadDPInference(CodeGenerator):
    
    def __init__(self,
                 model: OpenModel,
                 dp_rounds: int,
                 config: Dict[Text, Any],
                 use_vllm: bool = False):
    
        super().__init__()
    
        self.model = model
        self.dp_rounds = dp_rounds
        self.config = config
        self.use_vllm = use_vllm
    
    @overrides
    def inference(self,
                  dataloader: DataLoader,
                  save_path: str,
                  overwrite: bool = False):
        """Infernce on DP benchmark dataset in parallel threads.
        """
        if os.path.exists(save_path) and not overwrite:
            logger.warning(f"File {save_path} exists. To overwrite, please use --overwrite flag.")
            return

        if self.use_vllm:
            logger.info(f'Using VLLM for inference')
            config = SamplingParams(**self.config)
            records = self.vllm_inference(dataloader, config)
        else:
            logger.info(f'Using HF for inference')
            records = self.hf_inference(dataloader, self.config)

        with open(save_path, "w") as f:
            json.dump(records, f, indent=4)
    
    def vllm_inference(self, 
                       dataloader: DataLoader,
                       config: SamplingParams):
        records = []
        for batch in tqdm(dataloader, desc="Inferencing DP"):
            
            problem_statements: List[Text] = batch['inputs'] # batch_size x dp_rounds

            results = self.model.model.generate(problem_statements, 
                                                sampling_params=config)
            outputs = [result.outputs[0].text for result in results]

            problem_statements = [problem_statements[i:i+self.dp_rounds+1] for i in range(0, len(problem_statements), self.dp_rounds+1)]
            outputs = [outputs[i:i+self.dp_rounds+1] for i in range(0, len(outputs), self.dp_rounds+1)]

            for idx, (problem_statement, output) in enumerate(zip(problem_statements, outputs)):
                records.append({'problem_id': batch['problem_ids'][idx],
                                'problem_statements': problem_statement,
                                'outputs': output,
                                'constraints': batch['constraints'][idx]})
        
        return records
        
    def hf_inference(self, 
                     dataloader: DataLoader,
                     config: Dict[Text, Any]):
        with torch.no_grad():
            self.model.model.eval()
            records = []
            
            for batch in tqdm(dataloader, desc="Inferencing DP"):
                batch_inputs: Dict[Text, Tensor] = batch['inputs']
                batch_inputs = {k: v.to(torch.device('cuda')) for k, v in batch_inputs.items()}

                outputs = self.model.model.generate(**batch_inputs,
                                                    **config)
                outputs = outputs[:, batch['inputs']['input_ids'].shape[1]:]
                decoded_outputs = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                decoded_outputs = [decoded_outputs[i:i+self.dp_rounds+1] for i in range(0, len(decoded_outputs), self.dp_rounds+1)]
                
                del batch_inputs
                del outputs

                for idx, decoded_output in enumerate(decoded_outputs):
                    records.append({'problem_id': batch['problem_ids'][idx],
                                    'problem_statements': batch['problem_statements'][idx],
                                    'outputs': decoded_output,
                                    'constraints': batch['constraints'][idx]})
                
        return records