"""Download open-source models to local directory
"""
import argparse
import logging
import os
import json
import pandas as pd

def main(model_name, temp):
    logger = logging.getLogger(__name__)

    for p in ['basic', 'instructive', 'cot']:

        data_path = f'data/evaluations/temp_{temp}/{model_name}.json'
        logger.info(f'Opening file: {data_path}')

        with open(data_path, "r") as file:
            data = json.load(file)
        logger.info(f"JSON data for {f'text_{p}'} prompts loaded from {os.path.abspath(data_path)}")
        evaluations = [item["evaluation"][f'text_{p}'] for item in data]
        
        df = pd.DataFrame()
        df[f'{p}_evals'] = evaluations

        df[f'fluency']      = df[f'{p}_evals'].str.extract(r'(?i)Fluency:\s*(\d+)').astype('float')
        df[f'flexibility']  = df[f'{p}_evals'].str.extract(r'(?i)Flexibility:\s*(\d+)').astype('float')
        df[f'originality']  = df[f'{p}_evals'].str.extract(r'(?i)Originality:\s*(\d+)').astype('float')
        df[f'elaboration']  = df[f'{p}_evals'].str.extract(r'(?i)Elaboration:\s*(\d+)').astype('float')

        means = df[['elaboration', 'flexibility', 'fluency', 'originality']].mean().round(4)
        
        print(f'---\nResults from {model_name} {p}:\n---')
        print(f'Elaboration score: {means["elaboration"]}')
        print(f'Flexibility: {means["flexibility"]}')
        print(f'Fluency: {means["fluency"]}')
        print(f'Originality: {means["originality"]}')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Show results of evaluation on creative outputs from LLMs (using LLM-as-a-judge).')

    parser.add_argument('-model_name', 
                        type=str, 
                        default='Qwen2.5-72B-Instruct', 
                        help=f'Name of LLM being evaluated'
                        )
    
    parser.add_argument('-temp', 
                        type=int, 
                        default='1', 
                        )

    args = parser.parse_args()
    main(**vars(args)) 