from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
# from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch
import re
import os
from tqdm import tqdm  # Import tqdm for progress tracking
import subprocess

import sys
from api_wrapper import ModelWrapper

#### CONSTANTS ####
STORY_CONSTRAINTS_CSV = 'df3_updated_gpt_constraints.csv'
MAX_TOKENS = 1500
# MAX_MODEL_LEN = 8000


#### UTILS ####
import json
def load_json(filename):
    """
    Load a JSON file given a filename
    If the file doesn't exist, then return an empty dictionary instead
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
open_source_models = {
    "mistral_7b_instruct": {
        "hf_dir": "mistralai/Mistral-7B-Instruct-v0.3"
    },
    "llama3_8b_instruct": {
        "hf_dir": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "llama3_70b_instruct": {
        "hf_dir": "meta-llama/Llama-3.3-70B-Instruct",
        "max_model_len": 8000
    },
    "qwen_7b_instruct": {
        "hf_dir": "Qwen/Qwen2.5-7B-Instruct"
    },
    "qwen_32b_instruct": {
        "hf_dir": "Qwen/Qwen2.5-32B-Instruct",
        "max_model_len": 8000
    },
    "qwen_72b_instruct": {
        "hf_dir": "Qwen/Qwen2.5-72B-Instruct",
        "max_model_len": 8000
    },
    "qwen_72b_instruct_long": {
        "hf_dir": "Qwen/Qwen2.5-72B-Instruct",
        "max_model_len": 12000
    },
    "deepseek_llama_70b": {
        "hf_dir": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "max_model_len": 8000
    },
    "deepseek_qwen_32b": {
        "hf_dir": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "max_model_len": 8000
    },
    "deepseek_qwen_7b": {
        "hf_dir": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    },
    "olmo_7b": {
        "hf_dir": "allenai/OLMo-2-1124-7B-Instruct"
    },
    "olmo_13b": {
        "hf_dir": "allenai/OLMo-2-1124-13B-Instruct"
    },
    "olmo_13b_dpo": {
        "hf_dir": "allenai/OLMo-2-1124-13B-DPO"
    },
    "olmo_13b_sft": {
        "hf_dir": "allenai/OLMo-2-1124-13B-SFT"
    },
    "mistral_small_24b": {
        "hf_dir": "mistralai/Mistral-Small-24B-Instruct-2501"
    },
    "mixtral_8x7b": {
        "hf_dir": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }
}

def generate_response(tokenizer, olmo, prompt_text, max_tokens=MAX_TOKENS):
    # Load the model and tokenizer
    

    # Create chat history
    chat = [
        {"role": "user", "content": prompt_text},
    ]

    # Apply chat template and tokenize
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to('cuda')

    # Generate response
    # response = olmo.generate(input_ids=inputs.to(olmo.device), max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
    response = olmo.generate(input_ids=inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.95)
    inputs = inputs.cpu()
    # Decode and return response
    return tokenizer.batch_decode(response, skip_special_tokens=True)[0]

    
def clear_cache_if_needed(directory):
    # Execute the df command to check disk usage for the specified directory
    abs_directory = os.path.expanduser(directory)

    df_output = subprocess.check_output(['df', '-h', abs_directory]).decode('utf-8')
    # Split the output lines
    df_lines = df_output.split('\n')
    # Get the line containing usage information
    usage_line = df_lines[1]
    # Split the line into fields
    fields = usage_line.split()
    # Extract the usage percentage
    usage_percentage = int(fields[4].rstrip('%'))
    
    # Check if usage exceeds 85%
    files_in_directory = os.listdir('/home/rbheemreddy_umass_edu/.cache/huggingface/hub')
    print(files_in_directory)

    if usage_percentage>60:
        command = f"rm -rfv ~/.cache/huggingface/hub/*"
        subprocess.call(command, shell=True)

        print(f"Cache cleared successfully for models as usage percentage is {usage_percentage}")
    else:
        print("Disk usage is below 60%. No need to clear cache. Usage Percentage:", usage_percentage)

    
# Example usage:
directory = "~"  # Specify the directory you want to check (e.g., "~" for the home directory)


def clear_cuda_memory():
    torch.cuda.empty_cache()


def llm_batch_inference(
        llm, 
        all_prompt_data, 
        open_model = False, 
        sampling_params = None,
        config = {}
    ):
    '''
    - input: [{
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "messages": messages,
        "others": ...
    }]
    - output: [{
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "messages": messages,
        "others": ...,
        "raw_output": {generated_output}
    }]
    '''

    # llm = self.eval_model if parsing_model is None else parsing_model
    if open_model: # vllm case
        
        llm_outputs = llm.generate(
            [d['prompt_text'] for d in all_prompt_data],
            sampling_params
        )
        llm_outputs = [o.outputs[0].text for o in llm_outputs]
    else:
        llm_outputs = [
            llm.generate_response(
                messages = d['messages'],
                config = config
            )
            for d in tqdm(all_prompt_data)
        ]
        # print(llm_outputs[0])
        # exit(0)
    for i in range(len(all_prompt_data)):
        all_prompt_data[i]['FinalGeneratedStory'] = cleaning_story(llm_outputs[i])
    return all_prompt_data

def cleaning_story(raw_story):
    if "</think>" in raw_story:  # handle deepseek models
        raw_story = raw_story.split('</think>')[1]
    # match = re.search(r'[STORY - BEGIN]\s*([\s\S]*?)\s*[STORY - END]', raw_story, re.DOTALL)
    if '[STORY - BEGIN]' in raw_story:
        match = re.search(r'\[STORY - BEGIN\](.*?)\[STORY - END\]', raw_story, re.DOTALL)
    elif 'STORY - BEGIN' in raw_story:
        match = re.search(r'STORY - BEGIN(.*?)STORY - END', raw_story, re.DOTALL)
    else:
        match = None
    # print('=> in cleaning_story():')
    # print('\t=> raw_story:', raw_story)
    if match:
        cleaned_story = match.group(1).replace('[STORY - BEGIN]', '').replace('[STORY - END]', '').strip()
        # print('\t=> cleaned_story:', cleaned_story)
        return cleaned_story
    else:
        # print('\t=> no match found!!')
        return raw_story

def addNewStory(df, list_num_constraints, llm=None, config={}):
    """Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
    # Initialize an empty DataFrame to store the results
    single_instruction_df = pd.DataFrame(columns=['Instruction', 'Constraints', 'BaseStory', 'Direction', 'Model', 'SelectedConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])
    

    # if llm==None:
    #     raise NotImplementedError # we will use vllm for all models
    #     olmo = OLMoForCausalLM.from_pretrained(model).to('cuda')
    #     tokenizer = OLMoTokenizerFast.from_pretrained(model)
    test_size = config.get('test_size', 10e10)
    ## create prompt
    all_prompt_data = []
    for index, row in df.iterrows():
        prompt2_start = f"Now modify the existing story to accommodate the following constraints: {row['SelectedConstraints']} into the LLM generated story and come up with a new story in 500 words. To make your work clearer, mark the beginning of your modified story with '[STORY - BEGIN]' and the end with '[STORY - END]'"
        final_prompt = f"""User: "  {row['Instruction']}" \n BaseStory: " {row["BaseStory"]} " \n User Instruction: " {prompt2_start} """
        all_prompt_data.append({
            'Instruction': row['Instruction'],
            # 'Category': row['Category'],
            'Constraints': row['Constraints'],
            'BaseStory': row["BaseStory"],
            'Direction': row['Direction'],
            'Model': row["Model"],
            'SelectedConstraints': row['SelectedConstraints'],
            'Number_of_Constraints': row['Number_of_Constraints'],
            'prompt_text': final_prompt,
            'messages': [
                { 
                    "role": "user", 
                    "content": final_prompt
                }
            ]
            # 'FinalGeneratedStory': final_generated_story
        })
        if len(all_prompt_data) >= test_size: break
        # if index > 10: break # TODO remove it during production!!
    
    ## prompt llm 
    if llm is None: 
        open_model = False # api models
        llm = ModelWrapper(
            model_name = config['model_name'],
            api_key = config['api_key']
        )
    else:
        open_model = True # open-source models
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS, 
        temperature=config.get('temperature', 0.75), 
        top_p=1
    )
    
    all_prompt_data = llm_batch_inference(
        llm, 
        all_prompt_data, 
        open_model = open_model, 
        sampling_params = sampling_params,
        config = config
    )

    return pd.DataFrame(all_prompt_data)

    
    # for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"): 
    #     ## TODO: remove `head(10)` above before running real-experiments
    #     # print("Processing row number:", index)

    #     prompt2_start = f"Now modify the existing story to accommodate the following constraints: {row['SelectedConstraints']} into the LLM generated story and come up with a new story in 500 words: "
    #     final_prompt = f"""User: "  {row['Instruction']}" \n BaseStory: " {row["BaseStory"]} " \n User Instruction: " {prompt2_start} """
        
    #     if llm==None:
    #         raise NotImplementedError # we will use vllm for all models
    #         final_generated_story = generate_response(tokenizer, olmo, final_prompt, max_tokens)
    #     else:
    #         # sampling_params = 
    #         output2 = llm.generate([final_prompt], sampling_params)
    #         for output in output2:
    #             final_generated_story = output.outputs[0].text

    #     # Add the data to the result DataFrame

    #     single_instruction_df.loc[len(single_instruction_df)] = {
    #         'Instruction': row['Instruction'],
    #         # 'Category': row['Category'],
    #         'Constraints': row['Constraints'],
    #         'BaseStory': row["BaseStory"],
    #         'Direction': row['Direction'],
    #         'Model': row["Model"],
    #         'SelectedConstraints': row['SelectedConstraints'],
    #         'Number_of_Constraints': row['Number_of_Constraints'],
    #         'Final_Prompt': final_prompt,
    #         'FinalGeneratedStory': final_generated_story
    #     }
    #     # break ## 

    # return single_instruction_df


def generalcall(llm, config):
    # filename = "/home/rbheemreddy_umass_edu/vllm_trials/direction3/df3_gpt_12_50_selected_constraints.csv"

    # base_path = get_save_path(name_model=name_model)
    # os.makedirs(base_path, exist_ok = True)

    # filename = "/home/rbheemreddy_umass_edu/vllm_trials/Expansion/direction3/df3_updated_gpt_constraints.csv"
    auto_gen = pd.read_csv(STORY_CONSTRAINTS_CSV)
    unique_instructions = auto_gen['Instruction'].unique()

    auto_gen_eval = auto_gen[auto_gen['Instruction'].isin(unique_instructions)].copy()

    # Add new columns to store outputs
    auto_gen_eval['Final_Prompt'] = ''
    auto_gen_eval['FinalGeneratedStory'] = ''
    auto_gen_eval['Model'] = config['model_name']

    # List of constraints to try
    list_num_constraints = [7, 15, 23, 31, 39]

    # Initialize an empty list to store all generated DataFrames
    all_dfs = []
    count = 0

    combined_df = addNewStory(auto_gen_eval, list_num_constraints, llm, config=config)

    # Append the generated DataFrame to the list
    all_dfs.append(combined_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    total_stories_df = pd.concat(all_dfs, ignore_index=True)
    # print(total_stories_df.info())
    # Save the combined DataFrame to a single CSV file

    # if "direction3" in STORY_CONSTRAINTS_CSV:
    #     d = "d3"
    # elif "direction2" in STORY_CONSTRAINTS_CSV:
    #     d = 'd2'
    # d = "d3"
    # save_path = f"{base_path}/{d}_overnight_storygens_{base_path}_{d}.csv"
    # save_path = f"{base_path}"
    # os.makedirs(base_path, exist_ok=True)
    # print("Path saving file:", save_path)
    # total_stories_df.to_csv(os.path.join(save_path, f'{d}_{base_path}_{d}_{iter}.csv'), index=False)

    return total_stories_df


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('ERROR! Missing config file!')
        exit(1)
    config = load_json(sys.argv[1])
    print('=> config:', config)

    for model_config in config['experiments_list']:
        
        model = model_config['model_name']
        print("=> name of model", model)
        if model in open_source_models:
            hf_dir = open_source_models[model]["hf_dir"]
            max_model_len = open_source_models[model].get("max_model_len", None)
            llm = LLM(
                model=hf_dir, 
                dtype="bfloat16", 
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len = max_model_len 
            )
        else:
            llm = None 
            api_keys = load_json('api_keys.json')
            # print(api_keys)
            model_config['api_key'] = api_keys[model]
            model_config['temperature'] = 0.75
            model_config['top_p'] = 1
            if 'deepseek' not in model:
                model_config['max_tokens'] = MAX_TOKENS - 500
            else:
                model_config['max_tokens'] = MAX_TOKENS
        output_dir = './output/' + model_config['run_id']
        os.makedirs(output_dir, exist_ok=True)
        
        all_stories = []
        for i in range(1, 4):
            total_stories_df = generalcall(llm=llm, config=model_config)
            all_stories.append(total_stories_df['FinalGeneratedStory'])
            total_stories_df.to_csv(os.path.join(output_dir, f'{i}.csv'), index=False)

        total_stories_df['Story1'] = all_stories[0]
        total_stories_df['Story2'] = all_stories[1]
        total_stories_df['Story3'] = all_stories[2]
        total_stories_df = total_stories_df.drop(columns = ['FinalGeneratedStory'])
        total_stories_df.to_csv(os.path.join(output_dir, 'all_stories.csv'), index=False)

        print(f"=> Model {model} DONE")
        
        # Clear the model object from memory
        del llm
        del model
        import gc
        gc.collect()
        clear_cuda_memory()
        print("=> CUDA memory cleared.")


