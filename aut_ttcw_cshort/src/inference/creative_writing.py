from src.inference.inference_driver import InferenceDriver
from src.prompt_engineering.templates import creative_writing_inference_template
from src.prompt_engineering.functions import make_prompt
from src.utils.helpers import load_json, llm_batch_inference

class CreativeWritingInference(InferenceDriver):
    
    def __init__(self, config = {}):
        super().__init__(config) # set attributes

    def create_batched_prompt(self, input_data):
        '''
        input_data = [
            {
                "meta_data": {
                    "dataset": "creative_writing_eval",
                    "id": "{id}",
                    "eval_func": null
                },
                "input": {
                    "text": "Write a New Yorker-style story given the plot below. Make sure it is atleast {word_count} words. Directly start with the story, do not say things like "Here's the story [...]":",
                    "file": "",
                    "others": {
                        "plot": "{plot}"
                    }
                },
                "output": {
                    "content": "{url}"
                }
            },
            ...
        ]
        
        '''
        all_prompt_data = []
        test_size = self.config.get("test_size", -1)
        for dp in input_data:
            dp_prompt = make_prompt(
                {
                    "plot": dp["input"]["others"]['plot'].replace(
                        "{{word_count}}",
                        str(dp["input"]["others"]['avg_len'])
                    ),
                    "word_count": 2048 # TODO: change here
                },
                creative_writing_inference_template
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
        ## 1. load data and make prompt
        input_data_dir = self.config.get('input_data_dir', 'data/processed/ttcw/original.json')
        input_data = load_json(input_data_dir)
        if input_data == {}:
            print('[ERROR] Input not found!!')
            exit(-1)

        ## 2. make prompt
        all_prompt_data = self.create_batched_prompt(input_data)

        ## 3. run batched inference with llm
        if not self.use_open_model:
            self.config['max_tokens'] = 4096
            # self.config['temperature'] = 0.75
        llm_results = llm_batch_inference(
            self.llm, 
            all_prompt_data,
            self.use_open_model,
            self.sampling_params,
            self.config
        )
        print("len(llm_results):", len(llm_results))
        
        ## 4. run parsing 
        llm_results = self.parse_llm_outputs(llm_results)
        llm_results = dict(zip(
            [dp['prompt_id'] for dp in llm_results],
            llm_results
        ))
        # self.logger.info(llm_results)
        return llm_results

    def parse_llm_outputs(self, llm_results):
        # TODO
        return llm_results