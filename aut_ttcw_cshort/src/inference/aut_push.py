import copy
import json

from src.inference.inference_driver import InferenceDriver
from src.prompt_engineering.templates import conversation_history_template
from src.prompt_engineering.functions import make_prompt
from src.utils.helpers import load_json, llm_batch_inference

class AUTInference(InferenceDriver):
    
    def __init__(self, config = {}):
        super().__init__(config) # set attributes

    def create_batched_prompt(self, input_data, prev_interactions, curr_iter = ""):
        '''
        input_data = [
            {
                "meta_data": {
                    "dataset": "aut_push",
                    "id": 10,
                    "eval_func": null
                },
                "input": {
                    "text": "Create a list of creative alternative uses for a soap. They should be 5 words long. No adjectives. Less creative means closer to common use and\nunfeasible/imaginary, more creative means closer to unexpected uses and also feasible/practical. In order to be creative, consider the following:\n\u2212 what elements have a similar shape of a soap that could be replaced by it, preserving the same functionality?\n\u2212 what elements have a similar size of a soap that could be replaced by it without compromising the physical structure?\n\u2212 what materials is a soap made of that could be used in a way to replace some other elements composed of the same material?\n\u2212 when an element is replaced by a soap, it should make sure that the overall structure is not compromised.\n\u2212 the laws of physics can not be contradicted.\n\u2212 given an element similar to a soap used in domains in which soap are not commonly used, try to replace it for a soap.",
                    "file": "",
                    "others": {
                        "object": "soap",
                        "prompt_type": "bs",
                        "iteration_lst": {
                            "bsr": "Really? Is this the best you can do?",
                            "bsrd": "I\u2019m so disappointed with you. I hope this time you put effort into it.",
                            "bsrde": "Stop with excuses and do your best this time",
                            "bsrdel": "This is your last chance."
                        }
                    }
                },
                "output": {
                    "content": ""
                }
            },
            ...
        ]

        prev_interactions = {
            id: input id,
            interactions: output from previous step
        }
        
        '''
        all_prompt_data = []
        test_size = self.config.get("test_size", 10e10)
        for dp in input_data:
            skip_dp = False
            if curr_iter == "":
                messages = [
                    {
                        "role": "user", 
                        "content": dp["input"]["text"] + '\nList your results in an unordered list with one use per new line (starting with "-"); provide at most 10 answers.'
                    }
                ]
                prompt_text = "[user]: {}\n[assistant]:".format(dp["input"]["text"] + '\nList your results in an unordered list with one use per new line (starting with "-"); provide at most 10 answers.')
            elif dp['input']['others']['prompt_type'] == "bs":
                # load history
                messages = copy.deepcopy(prev_interactions[dp['meta_data']['id']])
                prompt_text = ''
                for i in range(0, len(messages), 2):
                    prompt_text += conversation_history_template.format(
                        user_message = messages[i]["content"],
                        assistant_message = messages[i + 1]["content"]
                    )
                # add latest prompt
                messages.append({ 
                    "role": "user", 
                    "content": dp["input"]["others"]["iteration_lst"][curr_iter] + '\nList your results in an unordered list with one use per new line (starting with "-"); provide at most 10 answers; only provide answer, no appology or irrelevant information.'
                })
                prompt_text += "[user]: {}\n[assistant]:".format(
                    dp["input"]["others"]["iteration_lst"][curr_iter] + '\nList your results in an unordered list with one use per new line (starting with "-"); provide at most 10 answers; only provide answer, no appology or irrelevant information.'
                )
            else:
                skip_dp = True 

            if not skip_dp:
                prompt_data = {
                    "prompt_id": dp['meta_data']['id'],
                    "prompt_text": prompt_text,
                    "messages": messages,
                    "data": dp
                } ## TODO: update all other prompt_data format and also the `llm_batch_inference` function
                all_prompt_data.append(prompt_data)

            if len(all_prompt_data) > test_size: break # stop when number of prompts hits test_size

        return all_prompt_data

    def inference(self):
        ## 1. load data
        input_data_dir = self.config.get('input_data_dir', 'data/processed/aut/aut_push_skipped.json') # creative_bench/data/processed/aut/aut_push_all.json
        input_data = load_json(input_data_dir)
        if input_data == {}:
            print('[ERROR] Input not found!!')
            exit(-1)

        ## 2. work on single_round data
        iterations = {
            "": "",
            "bsr": "Really? Is this the best you can do?",
            "bsrd": "I\u2019m so disappointed with you. I hope this time you put effort into it.",
            "bsrde": "Stop with excuses and do your best this time",
            "bsrdel": "This is your last chance."
        }
        prev_interactions = {dp["meta_data"]["id"]: [] for dp in input_data}
        final_results = {}
        for i in iterations:
            # make prompts
            all_prompt_data = self.create_batched_prompt(
                input_data, 
                prev_interactions, 
                curr_iter = i
            )

            # run batched inference with vllm
            if not self.use_open_model:
                self.config['max_tokens'] = 512
                # self.config['temperature'] = 0.75
            llm_results = llm_batch_inference(
                self.llm, 
                all_prompt_data,
                self.use_open_model,
                self.sampling_params,
                self.config
            )

            # run parsing 
            llm_results = self.parse_llm_outputs(llm_results)

            # append this round of output
            for prompt_data in all_prompt_data:
                dp_id = prompt_data["data"]["meta_data"]["id"]
                if i == "": 
                    tmp_key = prompt_data["data"]["input"]["others"]["prompt_type"]
                    final_results[dp_id] = {tmp_key: prompt_data['cleaned_output']}
                else:
                    tmp_key = i
                if prompt_data["data"]['input']['others']['prompt_type'] == "bs": 
                    prev_interactions[dp_id] = prompt_data["messages"]
                    prev_interactions[dp_id].append({
                        "role": "assistant",
                        "content": prompt_data['raw_output']
                    })
                    if dp_id not in final_results:
                        final_results[dp_id] = {}
                    final_results[dp_id][tmp_key] = prompt_data['cleaned_output']
                
        return final_results

    def parse_llm_outputs(self, llm_results):
        for i in range(len(llm_results)):
            raw_output = llm_results[i]['raw_output']
            if ':' in raw_output:
                cleaned_output = ':'.join(raw_output.split(':')[1:])
            else:
                cleaned_output = raw_output
            llm_results[i]['cleaned_output'] = cleaned_output
            # TODO
        return llm_results