import nltk
import re 

from src.inference.inference_driver import InferenceDriver
from src.prompt_engineering.templates import creative_short_story_template
from src.prompt_engineering.functions import make_prompt
from src.utils.helpers import load_json, llm_batch_inference

class CreativeShortStoryInference(InferenceDriver):
    
    def __init__(self, config = {}):
        super().__init__(config) # set attributes

    def create_batched_prompt(self, input_data):
        '''
        input_data = [
            {
                "meta_data": {
                    "dataset": "creative_short_story",
                    "id": "petrol-diesel-pump",
                    "eval_func": null
                },
                "input": {
                    "text": "",
                    "others": {
                        "items": ["petrol", "diesel", "pump"],
                        "pos": ["noun", "noun", "verb"],
                        "semantic_distance": "low",
                        "boring_theme": "going to the petrol station to pump diesel into a vehicle"
                    }
                },
                "output": {
                    "content": ""
                }
            }
            ...
        ]
        '''

        all_prompt_data = []
        test_size = self.config.get("test_size", -1)
        for dp in input_data:
            dp_prompt = make_prompt(
                {
                    "items": ', '.join(dp["input"]["others"]['items']),
                    "boring_theme": dp["input"]["others"]["boring_theme"]
                },
                creative_short_story_template
            )
            dp_messages = [{"role": "user", "content": dp_prompt}]
            prompt_data = {
                "prompt_id": dp['meta_data']['id'],
                "prompt_text": dp_prompt,
                "messages": dp_messages,
                "data": dp
            }
            all_prompt_data.append(prompt_data)
            if test_size > -1 and len(all_prompt_data) > test_size:
                break # stop when number of prompts hits test_size

        return all_prompt_data

    def inference(self):
        # return 
    
        ## 1. load data and make prompt
        input_data_dir = self.config.get('input_data_dir', 'data/processed/creative_short_story/all.json')
        input_data = load_json(input_data_dir)
        if input_data == {}:
            print('[ERROR] Input not found!!')
            exit(-1)

        ## 2. make prompt
        all_prompt_data = self.create_batched_prompt(input_data)

        ## 3. run batched inference with llm
        if not self.use_open_model:
            self.config['max_tokens'] = 256
            # self.config['temperature'] = 0.0
        llm_results = llm_batch_inference(
            self.llm, 
            all_prompt_data,
            self.use_open_model,
            self.sampling_params,
            self.config
        )
        
        ## 4. run parsing 
        llm_results = self.parse_llm_outputs(llm_results)
        llm_results = dict(zip(
            [dp['prompt_id'] for dp in llm_results],
            llm_results
        ))
        # self.logger.info(llm_results)
        return llm_results

    def parse_llm_outputs(self, llm_results):
        # sentence tokenizer 
        for dp in llm_results:
            # match = re.search(r"\[START\](.*?)\[END\]", dp['raw_output'], re.DOTALL)
            # print(match)
            # if match is not None:
            #     cleaned_output = match.group(1)
            # else:
            #     cleaned_output = dp['raw_output'] 
            raw_output = dp['raw_output']
            if '[END]' in raw_output:
                cleaned_output = raw_output.split('[END]')[0].replace('[START]', '')
            else:
                cleaned_output = raw_output
            # print(cleaned_output)
            cleaned_output = ' '.join(
                nltk.sent_tokenize(cleaned_output)[:10]
            )
            dp['cleaned_output'] = cleaned_output
        return llm_results
    
    