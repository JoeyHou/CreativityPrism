from src.evaluation.eval_driver import EvalDriver
from src.utils.helpers import load_json, write_json, llm_batch_inference, load_jsonl, write_jsonl
from src.prompt_engineering.templates import aut_evaluation_template, aut_evaluation_parsing_tempalte, aut_eval_single_use

# from vllm import SamplingParams, LLM
from tqdm import tqdm 
import numpy as np 
import pandas as pd 
# import torch 
# import openai
import re

class AUTEval(EvalDriver):

    def __init__(self, config = {}):
        EvalDriver.__init__(self, config)
        
    
    def create_batched_prompt(self, aut_results):
        eval_prompts = []
        counter = 0
        aut_demos = load_json('./data/processed/aut/aut_demos.json')
        aut_original_data = load_json('./data/processed/aut/aut_push_skipped.json')
        # print(aut_demos, aut_original_data)
        # print(aut_results)
        # print(aut_original_data)
        for dp_id in aut_results:
            for iter_id in aut_results[dp_id]:
                
                raw_output = aut_results[dp_id][iter_id]

                # handle deepseek models
                if "</think>" in raw_output: 
                    raw_output = raw_output.split('</think>')[1]
                
                # clean output text, remove non-char or non-num items and keep first 10
                output_use_list = [re.sub(r'[^A-Za-z0-9\s]', '', use).strip() for use in raw_output.split('\n')]
                output_use_list = [use for use in output_use_list if use != ''][:10]

                # look for the tool and demo data
                tool = ""
                for original_dp in aut_original_data:
                    if str(original_dp['meta_data']['id']) == str(dp_id):
                        tool = original_dp['input']['others']['object']
                        break 
                if tool == "": continue # skip some tools
                demo_data = aut_demos[tool]

                # use the new template 
                demo_use = '\n'.join([
                    aut_eval_single_use.format(use = use, score = score)
                    for use, score in demo_data
                ])
                output_use = '\n'.join([
                    aut_eval_single_use.format(use = use, score = "")
                    for use in output_use_list
                ])
                prompt_text = aut_evaluation_template.format(
                    tool = tool,
                    outputs = demo_use + '\n' + output_use
                )
                # + '\n' + aut_results[dp_id][iter_id]
                messages = [{
                    "role": "user", 
                    "content": prompt_text
                }]
                eval_prompts.append({
                    "prompt_id": str((dp_id, iter_id)),
                    "prompt_text": prompt_text,
                    "messages": messages,
                    "output_use_list": output_use_list
                })
                # self.logger.info(prompt_text)
                counter += 1
                # if counter > self.test_size: break 
            # if counter > self.test_size: break
        return eval_prompts

    def parse_llm_outputs(self, llm_outputs):
        '''
        - input: llm_outputs = [{
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "messages": messages,
            "others": ...,
            "raw_output": {generated_output}
        }]
        - output: llm_outputs = [{
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "messages": messages,
            "others": ...,
            "raw_output": {generated_output}, 
            "cleaned_output: ... <- this is what we want to get from this func
        }]
        '''
        for output_dp in llm_outputs:
            raw_output = output_dp['raw_output']
            output_lst = [line.strip() for line in raw_output.split('\n') if line.strip() != '\n']
            cleaned_output = []
            for use in output_dp['output_use_list']:
                for line in output_lst:
                    if use in line:
                        score = line.split(use)[1].replace(':', '')
                        try:
                            score = float(score)
                        except:
                            break 
                        cleaned_output.append([use, score])
                        break
            output_dp['cleaned_output'] = cleaned_output[:7] # pick the first 7 uses
        return llm_outputs

        # ## 1. setup parsing models 
        # if self.use_open_model:
        #     del self.llm # clear up the GRAM space 

        # parsing_model = LLM(
        #     model=self.local_parsing_model, 
        #     tensor_parallel_size=torch.cuda.device_count(), 
        #     dtype="bfloat16", 
        #     # max_model_len=self.parsing_output_len
        # )
        # parsing_sampling_params = SamplingParams(
        #     temperature = 0, 
        #     top_p = 1,
        #     max_tokens = self.parsing_output_len
        # )

        # ## 2. parsing via local llm
        # parsing_prompts = [
        #     {
        #         "prompt_text": aut_evaluation_parsing_tempalte.format(
        #             llm_output = out["raw_output"]
        #         )
        #     }
        #     for out in llm_outputs
        # ]
        # parsed_output = llm_batch_inference(
        #     parsing_model,
        #     parsing_prompts,
        #     open_model = True,
        #     sampling_params = parsing_sampling_params
        # )

        # ## 3. add result to original llm_outputs
        # for i in range(len(llm_outputs)):
        #     llm_outputs[i]["cleaned_output"] = parsed_output[i]["raw_output"]#.outputs[0].text

        # ## 4. extract scores
        # for out in llm_outputs:
        #     cleaned_output = []
        #     for line in out['cleaned_output'].split('\n'):
        #         line = line.strip()
        #         if '[parsed output - End]' in line: # ending 
        #             break 
        #         if '(Score:' in line: # found a score 
        #             use_case = line.split('(Score:')[0].replace('-', '').strip()
        #             use_score = line.split('(Score:')[1].strip()
        #             try:
        #                 use_score = float(use_score[0])
        #             except:
        #                 continue 
        #             if (use_case, use_score) not in cleaned_output:
        #                 cleaned_output.append((use_case, use_score))
        #     out['cleaned_output'] = cleaned_output
        # return llm_outputs

    def generate_eval_report(self, eval_outputs_cleaned):
        '''
        Generate a csv report for evaluation
        - input: eval_outputs_cleaned = [{
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "messages": messages,
            "others": ...,
            "raw_output": text_output,
            "cleaned_output: (use_case, use_score) <- this is what we care
        }]
        '''
        # 1. find all unique iteration ids
        unique_iter_id = []
        for dp in eval_outputs_cleaned:
            prompt_id = dp['prompt_id']
            if isinstance(prompt_id, str):
                dp_id, iter_id = prompt_id[1:-1].split(', ')
                dp['prompt_id'] = (
                    dp_id.replace('"', '').replace("'", ''), 
                    iter_id.replace('"', '').replace("'", '')
                )
                # dp['prompt_id'] = dp_id, iter_id
            iter_id = dp['prompt_id'][1]
            if iter_id not in unique_iter_id:
                unique_iter_id.append(iter_id)
        # print(unique_iter_id)

        # 2. accumulate average scores acorss those iterations 
        output_data = []
        for i in unique_iter_id:
            tmp_scores = []
            for dp in eval_outputs_cleaned:
                iter_id = dp['prompt_id'][1]
                # print(iter_id, i)
                if iter_id == i: # only include current iter id
                    tmp_scores.append(np.mean([
                        use_score for _, use_score in dp["cleaned_output"]
                    ]))
            # print('tmp_scores:', tmp_scores)
            tmp_scores = [s for s in tmp_scores if not np.isnan(s)]
            if len(tmp_scores) == 0:
                self.logger.info('no score in {}!'.format(i))
                continue 
            output_data.append({
                'setting': i,
                'avg': round(np.mean(tmp_scores), 2),
                'std': round(np.std(tmp_scores), 2),
            })
        return pd.DataFrame(output_data)
    
    def evaluation(self):
        '''
        - input: 
            aut_results = {
                dp_id: {
                    iteration_id: prompt_data['cleaned_output']
                }
            }

        - output:
            cleaned_eval_outputs = {

            }
        '''

        ## 0. load data 
        eval_output_raw_fp = 'data/output/{}/{}'.format(self.run_id, 'eval_output_raw.json')
        if self.config.get('eval_output_raw', False): # no existing eval_outputs_raw
            eval_output_raw = load_json(eval_output_raw_fp)
        else:
            aut_results = load_json('data/output/{}/{}'.format(
                self.config['run_id'], 
                'inference_output.json')
            )
            
            ## 1. prepare prompt
            eval_prompts = self.create_batched_prompt(aut_results)

            ## 2. batched eval
            eval_output_raw = llm_batch_inference(
                self.llm,
                eval_prompts,
                open_model = self.use_open_model,
                sampling_params = self.sampling_params
            )
            write_json(eval_output_raw, eval_output_raw_fp)
            

        ## 3. clean eval results
        eval_output_cleaned = self.parse_llm_outputs(eval_output_raw)
        # self.logger.info(eval_output_cleaned)

        ## 4. generate report 
        eval_report = self.generate_eval_report(eval_output_cleaned)
        
        return eval_report, eval_output_cleaned #, eval_output_raw
    
