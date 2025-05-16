from src.evaluation.eval_driver import EvalDriver
from src.utils.helpers import load_json, llm_batch_inference, write_json
from src.prompt_engineering.templates import creative_writing_evaluation_template, creative_writing_evaluation_fewshot

from tqdm import tqdm 
import numpy as np 
import pandas as pd 
from vllm import LLM
# import openai
# import math


ttwc_question_json_fp = './data/processed/ttcw/ttcw_questions.json'
ttcw_few_shot_fp = './data/processed/ttcw/ttcw_few_shot.json'
all_stories_fp = './data/processed/ttcw/all_stories.json'
selected_ttcw_index = [1, 2, 6, 13]

class CreativeWritingEval(EvalDriver):
    
    def __init__(self, config = {}):
        EvalDriver.__init__(self, config)
        
    # DONE
    def create_batched_prompt(self, creative_writing_results):
        '''
        - input: 
            creative_writing_results = {
                dp_id: {
                    "id": dp_id,
                    "prompt": inference_prompt,
                    "data": dp_data,
                    "raw_output": raw_output
                }
            }

        - output:
            eval_prompts = [
                {
                    "prompt_id": (dp_id, ttwc_q_id),
                    "prompt": eval_prompt
                }
            ]
        '''
        eval_prompts = []
        test_size = self.config.get("test_size", 10e10)
        
        # loading related data 
        ttwc_questions = load_json(ttwc_question_json_fp) # ttwc questions
        few_shot_examples = load_json(ttcw_few_shot_fp)
        all_stories = load_json(all_stories_fp)

        # create prompt 
        for dp_id in creative_writing_results:
            for ttcw_q in ttwc_questions:
                # gather output info 
                story = creative_writing_results[dp_id]["raw_output"]
                
                if "</think>" in story: # handle deepseek models
                    story = story.split('</think>')[1]
                    
                full_prompt = ttcw_q['full_prompt']
                ttcw_q_id = ttcw_q['ttcw_idx']
                if ttcw_q_id not in selected_ttcw_index: continue # skip non-selected questions 

                # create prompt for evaluation 
                ## construct few-shot demostrations
                if self.config.get('few_shot', False):
                    few_shot_demo = ''
                    pos_demo = few_shot_examples['pos_data'][str(ttcw_q_id)]
                    few_shot_demo += creative_writing_evaluation_fewshot.format(
                        story = all_stories[pos_demo['story_id']],
                        answer = 'Yes',
                        exp = pos_demo['explanation']
                    ) + '\n'
                    neg_demo = few_shot_examples['neg_data'][str(ttcw_q_id)]
                    few_shot_demo += creative_writing_evaluation_fewshot.format(
                        story = all_stories[neg_demo['story_id']],
                        answer = 'No',
                        exp = neg_demo['explanation']
                    ) + '\n'
                else:
                    few_shot_demo = ''
                
                ## change the prompt to likert scale 
                if self.config.get('5_scale', False):
                    full_prompt = full_prompt.replace(
                        "between 'Yes' or 'No' only",
                        "as a score between 1 and 5, with 1 being defintely No, 5 being defintely Yes, and scores in between as answers between yes and no, propotionally."
                    )
                    template = creative_writing_evaluation_template.replace(
                        '"**Answer**: [Yes/No]"',
                        '"**Answer**: [Score]"'
                    ).replace(
                        'binary (Yes/No)',
                        ''
                    )
                else:
                    template = creative_writing_evaluation_template

                ## construct the prompt 
                prompt_text = template.format(
                    story = story,
                    full_prompt = full_prompt,
                    few_shot_demo = few_shot_demo
                )
                messages = [{
                    "role": "user", 
                    "content": prompt_text
                }]

                # add to eval_prompts
                eval_prompts.append({
                    "prompt_id": (dp_id, ttcw_q_id),
                    "prompt_text": prompt_text,
                    "messages": messages
                })
            # self.logger.info('=> len(eval_prompts): ' + str(len(eval_prompts)))
            # if len(eval_prompts) > test_size: break 
        # exit(1)
        return eval_prompts

    def parse_llm_outputs(self, llm_results):
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
            "cleaned_output: 0/1
        }]
        '''
        if self.config.get('5_scale', False):
            for dp in llm_results:
                clipped_output = dp['raw_output']

                clipped_output = clipped_output.split('**Answer**:')
                if len(clipped_output) >= 2:
                    clipped_output = clipped_output[1]
                else:
                    clipped_output = clipped_output[0]
                clipped_output = clipped_output.strip()[:10].lower()
                
                try:
                    dp['cleaned_output'] = int(clipped_output[0])
                except:
                    dp['cleaned_output'] = -1
            return llm_results
        else:
            for dp in llm_results:
                clipped_output = dp['raw_output'].strip()[:20].lower()
                if 'yes' in clipped_output:
                    dp['cleaned_output'] = 1
                else:
                    dp['cleaned_output'] = 0
            return llm_results

    def generate_eval_report(self, eval_output_cleaned):
        '''
        Generate a csv report for evaluation
        - input: eval_output_cleaned = [{
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "messages": messages,
            "others": ...,
            "raw_output": {generated_output},
            "cleaned_output: ...
        }]
        '''
        ttwc_questions = load_json(ttwc_question_json_fp) # ttwc questions
        all_scores = {}
        for prompt_dp in eval_output_cleaned:
            dp_id, ttwc_q_id = prompt_dp["prompt_id"]
            if dp_id not in all_scores:
                all_scores[dp_id] = {'dp_id': dp_id}
            ttwc_key = [ 
                q['torrance_dimension'] + ' - ' + q['category']
                for q in ttwc_questions if q['ttcw_idx'] == int(ttwc_q_id)
            ][0]
            all_scores[dp_id][ttwc_key] = prompt_dp['cleaned_output']
        # self.logger.info(str(list(all_scores.values())))
        return pd.DataFrame(list(all_scores.values()))
    
    def evaluation(self):
        '''
        - input: 
            {
                dp_id: {
                    "id": dp_id,
                    "prompt": inference_prompt,
                    "data": dp_data,
                    "raw_output": raw_output
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
            creative_writing_results = load_json('data/output/{}/{}'.format(
                self.config['run_id'], 
                'inference_output.json')
            )
            ## 1. prepare prompt
            eval_prompts = self.create_batched_prompt(creative_writing_results)

            ## 2. batched eval
            eval_output_raw = llm_batch_inference(
                self.llm,
                eval_prompts,
                open_model = self.use_open_model,
                sampling_params = self.sampling_params
            )
            write_json(eval_output_raw, eval_output_raw_fp) 
        
        # print(eval_output_raw[:1])

        ## 3. clean eval results
        eval_output_cleaned = self.parse_llm_outputs(eval_output_raw)
        # self.logger.info(eval_output_cleaned)

        ## 4. generate report 
        eval_report = self.generate_eval_report(eval_output_cleaned)
        
        return eval_report, eval_output_cleaned #, eval_output_raw
    


