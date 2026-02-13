from src.evaluation.eval_driver import EvalDriver
from src.evaluation.story_metrics import (
    compute_dsi,
    compute_surprise,
    compute_n_gram_diversity,
    compute_inverse_homogenization,
    compute_novelty,
    compute_theme_uniqueness
)
from src.utils.helpers import load_json, llm_batch_inference, write_json
from src.prompt_engineering.templates import creative_writing_evaluation_template, creative_writing_evaluation_fewshot

import pandas as pd 
import numpy as np

class CreativeShortStoryEval(EvalDriver):
    
    def __init__(self, config = {}):
        EvalDriver.__init__(self, config)
        
    def create_batched_prompt(self, creative_writing_results):
        return 

    def parse_llm_outputs(self, llm_results):
        return 

    def generate_eval_report(self, eval_output_cleaned):
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

        '''
        # self.logger.info(eval_output_cleaned)
        avg_dsi = round(np.mean([dp['eval_result']['dsi'] for dp in eval_output_cleaned.values()]), 4)
        avg_sur = round(np.mean([dp['eval_result']['surprise'] for dp in eval_output_cleaned.values()]), 4)
        avg_ngram_diversity = {
            '{}_gram'.format(i): round(np.mean([
                dp['eval_result']['n_gram_diversity'][i] for dp in eval_output_cleaned.values()
            ]), 2)
            for i in range(len(
                list(eval_output_cleaned.values())[0]['eval_result']['n_gram_diversity']
            )) 
        }
        avg_ngram_diversity['avg_dsi'] = avg_dsi
        avg_ngram_diversity['avg_sur'] = avg_sur
        # self.logger.info(self.model_name)
        # self.logger.info(avg_ngram_diversity)
        texts = [dp['raw_output'] for dp in eval_output_cleaned.values()]
        avg_ngram_diversity['inverse_homogenization'] = np.mean(compute_inverse_homogenization(texts))
        avg_ngram_diversity['novelty'] = np.mean(compute_novelty(texts))
        avg_ngram_diversity['theme_uniqueness'] = np.mean(compute_theme_uniqueness(texts))
        
        return pd.DataFrame([avg_ngram_diversity])

    
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
        inference_results = load_json('data/output/{}/{}'.format(
            self.config['run_id'], 
            'inference_output.json')
        )
        for dp_id in inference_results:
            dp = inference_results[dp_id]
            n_gram_diversity, all_n_gram_freqs = compute_n_gram_diversity(dp['raw_output'])
            # semantic_diversity = compute_inverse_homogenization(dp['raw_output'])
            surprises, raw_surprises = compute_surprise(dp['raw_output'])
            dp['eval_result'] = {
                'dsi': compute_dsi(dp['raw_output']),
                'surprise': surprises,
                'n_gram_diversity': n_gram_diversity
            }
            # self.logger.info(str(dp['eval_result']))
        eval_output_cleaned = inference_results
        # print(eval_output_cleaned)
        eval_report = self.generate_eval_report(eval_output_cleaned)
        return eval_report, eval_output_cleaned