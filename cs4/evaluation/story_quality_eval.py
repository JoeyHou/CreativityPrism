import argparse
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime

from vllm import SamplingParams, LLM
import torch
import gc
import re
import json 
import copy 
from tqdm import tqdm 

GPT_MODEL = "gpt-4.1-mini"
# GPT_MODEL = 'gpt-3.5-turbo'
vLLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

###### prompt template #######
system_prompt1 = """
You are an English writing expert and you can compare and evaluate story essays on these metrics with the following definitions -
    1. Grammar: Which story has better writing and grammar comparitively?
    2. Coherence: Which story has a better logical flow and the writing fits together with respect to the plot?
    3. Likability: Which story do you find more enjoyable to read?
You will be given two Stories - Story A and Story B.
Add a rating out of 5 for each category, specify which story you prefer for each metric by responding with just the letter "A" or "B" followed by a hyphen and one line reasoning for your preference.
For each category provide a category winner story as the letter "A" or "B", based on the category ratings.
Finally, assign an overall winner story as the letter "A" or "B" based on the ratings and category wins.
(if an story is empty, give it zero scores)

IMPORTANT - DO NOT GIVE ANY OTHER TEXT APART FROM THE SCORE, METRICS AND PREFERENCE. FOLLOW THE EXACT FORMAT AS GIVEN IN THE FOLLOWING EXAMPLES.

EXAMPLE OUTPUT 1:
Grammar  Preference: A
A - 5/5: Story A has a few minor grammatical issues, but overall, it demonstrates strong control of language.
B - 4/5: Story B is well-written but has slightly more noticeable issues in grammar and sentence structure.
Coherence  Preference: A
A - 4.5/5: Story B has a strong coherence, effectively conveying the emotional journey and the progression of events.
B - 4/5: Story A maintains a consistent and engaging narrative flow, though some parts are a bit abstract.
Likability  Preference: A
A - 4/5: Story B's realistic and emotional narrative is likely to resonate more with a wide range of readers.
B - 3.5/5: Story A is imaginative and intriguing, but its abstract nature might not appeal to all readers.
Overall Winner: A

EXAMPLE OUTPUT 2:
Grammar Preference: B
A - 3/5: Story A has some complex sentences that are difficult to follow, with occasional grammatical errors.
B - 4/5: Story B is well-written with minor grammatical mistakes and clear sentence structures.
Coherence Preference: B
A - 2/5: The plot of Story A is somewhat confusing and disjointed, especially with the sudden introduction of an old sage.
B - 5/5: Story B maintains a coherent narrative, with each event logically building on the previous one, enhancing the story's flow.
Likability Preference: B
A - 3/5: Story A is heartfelt but its erratic narrative structure detracts from its overall appeal.
B - 5/5: Story B is compelling and maintains consistent character development, making it more enjoyable and engaging.
Overall Winner: B

"""

prompt0 = """
Story A:
{story1}

Story B:
{story2}

SCORE OUTPUT:
"""

parsing_prompt = '''
### EXAMPLE 1
[INPUT]
Grammar  Preference: A
A - 5/5: Story A has a few minor grammatical issues, but overall, it demonstrates strong control of language.
B - 4/5: Story B is well-written but has slightly more noticeable issues in grammar and sentence structure.
Coherence  Preference: A
A - 4.5/5: Story B has a strong coherence, effectively conveying the emotional journey and the progression of events.
B - 4/5: Story A maintains a consistent and engaging narrative flow, though some parts are a bit abstract.
Likability  Preference: A
A - 4/5: Story B's realistic and emotional narrative is likely to resonate more with a wide range of readers.
B - 3.5/5: Story A is imaginative and intriguing, but its abstract nature might not appeal to all readers.
Overall Winner: A
[OUTPUT]
JSON - BEGIN
{
    "grammar_score_A": 5,
    "grammar_score_B": 4,
    "coherence_score_A": 4.5,
    "coherence_score_B": 4,
    "likability_score_A": 4,
    "likability_score_B": 3.5,
    "grammar_pref": "A",
    "coherence_pref": "A",
    "likability_pref": "A",
    "overall_pref": "A"
}
JSON - END

### EXAMPLE 2
[INPUT]
Grammar Preference: B
A - 3/5: Story A has some complex sentences that are difficult to follow, with occasional grammatical errors.
B - 4/5: Story B is well-written with minor grammatical mistakes and clear sentence structures.
Coherence Preference: B
A - 2/5: The plot of Story A is somewhat confusing and disjointed, especially with the sudden introduction of an old sage.
B - 5/5: Story B maintains a coherent narrative, with each event logically building on the previous one, enhancing the story's flow.
Likability Preference: B
A - 3/5: Story A is heartfelt but its erratic narrative structure detracts from its overall appeal.
B - 5/5: Story B is compelling and maintains consistent character development, making it more enjoyable and engaging.
Overall Winner: B
[OUTPUT]
JSON - BEGIN
{
    "grammar_score_A": 3,
    "grammar_score_B": 4,
    "coherence_score_A": 2,
    "coherence_score_B": 5,
    "likability_score_A": 3,
    "likability_score_B": 5,
    "grammar_pref": "B",
    "coherence_pref": "B",
    "likability_pref": "B",
    "overall_pref": "B"
}
JSON - END

Now converting the following input into json format, with "JSON - BEGIN" at the start and "JSON - END" at the end.
[INPUT]
VLLM_OUTPUT
[OUTPUT]'''


# Initialize OpenAI client
def initialize_openai(api_key):
    return OpenAI(api_key=api_key)

# Chat function to send prompt to OpenAI API
def chat(client, instruction, model=GPT_MODEL, system_prompt=""):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
    )

    return response

# Parse the evaluation results
def parse_evaluation(evaluation):
    if pd.isna(evaluation):
        return None
    try:
        lines = [line.strip() for line in evaluation.split('\n') if line.strip()]
        parsed = {
            'grammar_score_A': float(lines[1].split('-')[1].split('/')[0].strip()),
            'grammar_score_B': float(lines[2].split('-')[1].split('/')[0].strip()),
            'coherence_score_A': float(lines[4].split('-')[1].split('/')[0].strip()),
            'coherence_score_B': float(lines[5].split('-')[1].split('/')[0].strip()),
            'likability_score_A': float(lines[7].split('-')[1].split('/')[0].strip()),
            'likability_score_B': float(lines[8].split('-')[1].split('/')[0].strip()),
            'grammar_pref': lines[0].split(': ')[1],
            'coherence_pref': lines[3].split(': ')[1],
            'likability_pref': lines[6].split(': ')[1],
            'overall_pref': lines[9].split(': ')[1]
        }
        a = 0
        b = 0
        if parsed['grammar_score_A'] >= parsed['grammar_score_B']:
            parsed['grammar_pref'] = "A"
            a += 1
        else:
            parsed['grammar_pref'] = "B"
            b += 1

        if parsed['coherence_score_A'] >= parsed['coherence_score_B']:
            parsed['coherence_pref'] = "A"
            a += 1
        else:
            parsed['coherence_pref'] = "B"
            b += 1

        if parsed['likability_score_A'] >= parsed['likability_score_B']:
            parsed['likability_pref'] = "A"
            a += 1
        else:
            parsed['likability_pref'] = "B"
            b += 1

        if a >= b:
            parsed['overall_pref'] = "A"
        else:
            parsed['overall_pref'] = "B"

        return parsed
    except Exception as e:
        print(e)
        return None

def pairwise_eval(client, story1, story2, model=GPT_MODEL):
    # Prompts
    response1 = chat(client, prompt0, model=model, system_prompt=system_prompt1)
    return response1.choices[0].message.content

# Evaluate stories and save results
def evaluate_stories(grouped_dfs, client, output_dir, max_trials=35, max_redo=3):
    count = 0
    sampling_params = SamplingParams(
        temperature = 0,
        top_p = 1,
        max_tokens = 512
    )
    if client is None:
        llm = LLM(
            vLLM_MODEL, 
            #"Qwen/Qwen2.5-72B-Instruct", 
            # "meta-llama/Llama-3.3-70B-Instruct",
            dtype = 'bfloat16',
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len = 10000
        )
    else:
        llm = None 

    for instruction, df in grouped_dfs.items():
        count += 1
        # if count > max_trials:
        #     continue
        # df = copy.deepcopy(df.head(20))            
        if client is None:
            ### prep prompts ###
            all_prompt_data = []
            if 'midstory' in df.columns:
                print('=> running evaluator check! midstory found in df!')
                df['midstory'] = df['midstory'].apply(lambda x: x.replace('!@#$', '\n'))
                df['Story1'] = df['Story1'].apply(lambda x: x.replace('!@#$', '\n'))
            for index, other_row in df.iterrows():
                rand_trial = np.random.randint(2) # randomly assign order
                if 'midstory' in df.columns:
                    story_a, story_b = (other_row['Story1'], other_row['midstory']) if rand_trial else (other_row['midstory'], other_row['Story1'])
                else:
                    row_with_11 = df[df['Number_of_Constraints'] == 23].iloc[1]
                    story_a, story_b = (other_row, row_with_11) if rand_trial else (row_with_11, other_row)
                    story_a, story_b = story_a['Story1'], story_b['Story1']
                results = None
                needs_parsing = -1

                all_prompt_data.append({
                    "stories": (story_a, story_b),
                    "order": rand_trial,
                    "needs_parsing": needs_parsing
                })
                
            ### prompt vllm models ###
            prompt_lst = [
                '[System Prompt] {}\n\n[Stories]\n'.format(
                    system_prompt1,
                ) + prompt0.format(
                    story1 = dp['stories'][0],
                    story2 = dp['stories'][1]
                )
                for dp in all_prompt_data
            ]
            # print('\n\nprompt examples:', prompt_lst[0], '\n\n')
            # idx_to_run = list(range(len(prompt_lst)))
            # all_output = {
            #     i: '[FAILED TO GET RESULT]'
            #     for i in range(len(prompt_lst))
            # }
            # new_idx_to_run = idx_to_run
            # iter_ran = 0
            # for _ in range(max_redo):
            #     if len(new_idx_to_run) == 0: break # if there is nothing left, stop!
            #     tmp_prompt_lst = [prompt_lst[i] for i in idx_to_run]
            #     response_lst = [
            #         out.outputs[0].text
            #         for out in llm.generate(tmp_prompt_lst, sampling_params)
            #     ]
            #     new_idx_to_run = []
            #     for j in range(len(response_lst)):
            #         res = response_lst[j]
            #         match = re.search('Overall Winner: [A,B]', res)
            #         if match:
            #             all_output[idx_to_run[j]] = res 
            #         else:
            #             new_idx_to_run.append(idx_to_run[j])
            #     iter_ran += 1
            # print('=> Total iteration:', iter_ran)
            # response_lst = list(all_output.values())

            response_lst = [
                out.outputs[0].text
                for out in llm.generate(prompt_lst, sampling_params)
            ]
            raw_responses = [
                res for res in response_lst
            ]
            response_lst = [
                res[:res.index('Overall Winner:') + 40] if 'Overall Winner:' in res else res
                for res in response_lst 
            ]

            ### post processing ### 
            #### Version 2 - current ####
            parsing_prompt_lst = [
                parsing_prompt.replace('VLLM_OUTPUT', vllm_output)
                for vllm_output in response_lst
            ]
            # counter = 0
            special_keys = ["needs_parsing", "order"]
            all_json = []
            parsed_output = llm.generate(
                parsing_prompt_lst, 
                SamplingParams(temperature = 0, max_tokens = 512)
            )

            for i in range(len(parsed_output)):

                # first try naive parsing 
                naive_parsing = parse_evaluation(response_lst[i])
                if naive_parsing is not None:
                    all_json.append(naive_parsing)
                    all_prompt_data[i].update(naive_parsing)
                    for key in naive_parsing:
                        if key not in special_keys:
                            special_keys.append(key)
                    all_prompt_data[i]["needs_parsing"] = 0
                    continue # stop here 
                    
                # then try json parsing with vllm
                match = re.search(r'JSON - BEGIN\s*([\s\S]*?)\s*JSON - END', parsed_output[i].outputs[0].text, re.DOTALL)
                if match:
                    json_content = match.group(1).replace('JSON - BEGIN', '').replace('JSON - END', '').strip()
                    try:
                        tmp_data = json.loads(json_content)
                        all_prompt_data[i].update(tmp_data)
                        for key in tmp_data:
                            if key not in special_keys:
                                special_keys.append(key)
                        all_prompt_data[i]["needs_parsing"] = 0
                    except:
                        tmp_data = ('[JSON PARSING ERROR]\n original output:' + parsed_output[i].outputs[0].text)
                        all_prompt_data[i]["needs_parsing"] = 2
                else:
                    tmp_data = ('[RE PARSING ERROR]\n original output:' + parsed_output[i].outputs[0].text)
                all_json.append(tmp_data)

            #### Version 1 ####
            # for out in parsed_output:
            #     match = re.search(r'JSON - BEGIN\s*([\s\S]*?)\s*JSON - END', out.outputs[0].text, re.DOTALL)
            #     if match:
            #         json_content = match.group(1).replace('JSON - BEGIN', '').replace('JSON - END', '').strip()
            #         # print("json found!")
            #         try:
            #             tmp_data = json.loads(json_content)
            #             all_prompt_data[counter].update(tmp_data)
            #             for key in tmp_data:
            #                 if key not in special_keys:
            #                     special_keys.append(key)
            #             all_prompt_data[counter]["needs_parsing"] = 0
            #         except:
            #             tmp_data = ('[JSON PARSING ERROR]')
            #             all_prompt_data[counter]["needs_parsing"] = 2
            #         all_json.append(tmp_data)
            #     else:
            #         all_prompt_data[counter]["needs_parsing"] = 1
            #         naive_parsing = parse_evaluation(response_lst[counter])
            #         if naive_parsing is None:
            #             all_json.append('[RE MATCHING ERROR]' + out.outputs[0].text)
            #         else:
            #             all_json.append(naive_parsing)
            #             # all_json.append(tmp_data)
            #             all_prompt_data[counter].update(naive_parsing)
            #             for key in naive_parsing:
            #                 if key not in special_keys:
            #                     special_keys.append(key)
            #             all_prompt_data[counter]["needs_parsing"] = 0
            #         # all_json.append('[RE MATCHING ERROR]' + out.outputs[0].text)
            #     counter += 1

            #### Version 3 ####
            # special_keys = ["needs_parsing", "order"]
            # all_json = {}
            # # first try naive parsing 
            # vllm_parsing_id = []
            # for i in range(len(response_lst)):
            #     # first try naive parsing 
            #     naive_parsing = parse_evaluation(response_lst[i])
            #     if naive_parsing is not None:
            #         all_json[i] = (naive_parsing)
            #         all_prompt_data[i].update(naive_parsing)
            #         for key in naive_parsing:
            #             if key not in special_keys:
            #                 special_keys.append(key)
            #         all_prompt_data[i]["needs_parsing"] = 0
            #     else:
            #         vllm_parsing_id.append(i)
            # print('=> stage 1 parsing success:', len(response_lst) - len(vllm_parsing_id))
            
            # # then vllm parsing
            # parsing_prompt_lst = [
            #     parsing_prompt.replace('VLLM_OUTPUT', response_lst[i])
            #     for i in vllm_parsing_id
            # ]
            # # counter = 0
            
            # parsed_output = llm.generate(
            #     parsing_prompt_lst, 
            #     SamplingParams(temperature = 0, max_tokens = 512)
            # )

            # for i in range(len(parsed_output)):
            #     original_indx = vllm_parsing_id[i]
            #     match = re.search(r'JSON - BEGIN\s*([\s\S]*?)\s*JSON - END', parsed_output[i].outputs[0].text, re.DOTALL)
            #     if match:
            #         json_content = match.group(1).replace('JSON - BEGIN', '').replace('JSON - END', '').strip()
            #         try:
            #             tmp_data = json.loads(json_content)
                        
            #             all_prompt_data[original_indx].update(tmp_data)
            #             for key in tmp_data:
            #                 if key not in special_keys:
            #                     special_keys.append(key)
            #             all_prompt_data[original_indx]["needs_parsing"] = 0
            #         except:
            #             tmp_data = ('[JSON PARSING ERROR]\n original output:' + parsed_output[i].outputs[0].text)
            #             all_prompt_data[original_indx]["needs_parsing"] = 2
            #     else:
            #         tmp_data = ('[RE PARSING ERROR]\n original output:' + parsed_output[i].outputs[0].text)
            #     all_json[original_indx] = (tmp_data)
            
            # ## re-order the list 
            # all_json = [all_json[idx] for idx in range(len(response_lst))]
            
            # print("all_prompt_data", all_prompt_data)
            for key in special_keys:
                df[key] = [dp[key] if key in dp else None for dp in all_prompt_data ]
            df['evaluations'] = response_lst
            df['all_json'] = all_json
            df['raw_responses'] = raw_responses
            df['raw_parsed_output'] = [out.outputs[0].text for out in parsed_output]
            
            ### cleaning up ###
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            print("=> CUDA memory cleared.")
        ## Openai method
        else:
            # raise NotImplementedError
            for index, other_row in tqdm(df.iterrows(), total=df.shape[0]):
                # if other_row['story_id'] == row_with_11['story_id']:
                #     continue
                
                rand_trial = np.random.randint(2) # randomly assign order
                if 'midstory' in df.columns:
                    df['midstory'] = df['midstory'].apply(lambda x: x.replace('!@#$', '\n'))
                    df['Story1'] = df['Story1'].apply(lambda x: x.replace('!@#$', '\n'))
                    story_a, story_b = (other_row['Story1'], other_row['midstory']) if rand_trial else (other_row['midstory'], other_row['Story1'])
                else:
                    # row_with_11 = df[df['Number_of_Constraints'] == 23].iloc[1]
                    # story_a, story_b = (other_row, row_with_11) if rand_trial else (row_with_11, other_row)
                    # story_a, story_b = story_a['Story1'], story_b['Story1']
                    midstory = get_mid_story(df)
                    curr_story = other_row['Story1']
                    story_a, story_b = (curr_story, midstory) if rand_trial else (midstory, curr_story)

                results = None
                needs_parsing = -1
                
                for attempt in range(max_redo):
                    try:
                        results = pairwise_eval(client, story_a, story_b)
                        parsed_results = parse_evaluation(results)
                        if parsed_results:
                            for key, value in parsed_results.items():
                                df.loc[other_row.name, key] = value
                            df.loc[other_row.name, 'order'] = rand_trial
                            needs_parsing = 0
                            break
                    except Exception as e:
                        print(f"Error during evaluation: {e}")
                        needs_parsing = 1
                        break

                df.loc[other_row.name, 'needs_parsing'] = needs_parsing
                df.loc[other_row.name, 'evaluations'] = results
            
        print(f"Evaluations complete for instruction {instruction}")

        # Save the DataFrame to CSV
        output_file = os.path.join(output_dir, "eval_quality.csv")
        df.to_csv(output_file, index=False)
        grouped_dfs[instruction] = df

def get_mid_story(df):
    row_with_11 = df[df['Number_of_Constraints'] == 23].iloc[1]
    if row_with_11['Story1'].strip() != '':
        return row_with_11['Story1']

    tmp_df = df[df['Number_of_Constraints'] == 23]
    for i in range(tmp_df.shape[0]):
        row_with_11 = df[df['Number_of_Constraints'] == 23].iloc[i]
        for story in ['Story1', 'Story2', 'Story3']:
            if row_with_11[story].strip() != '':
                return row_with_11[story]
    return ''

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description="Run story evaluation using OpenAI API")
    
    # API key and input/output paths
    parser.add_argument("--use_gpt", action="store_true")
    # parser.add_argument("--input_file", required=True, help="Path to input CSV file")
    # parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_trials", type=int, default=35, help="Maximum number of trials for evaluation")
    
    args = parser.parse_args()

    # Initialize OpenAI API client
    if args.use_gpt:
        api_key = os.environ['OPENAI_API_KEY']
        client = initialize_openai(api_key)
    else:
        client = None # by default, we use local model!
        
    # Load the input file into a pandas DataFrame
    story_dir = './story_generator/output'
    df = pd.read_csv("{}/{}/all_stories.csv".format(
        story_dir, args.run_id
    ))

    print('=> Done reading all_stories.csv!')
    
    # Group the DataFrame by instruction if needed
    grouped_dfs = {"default": df}  # In case grouping is needed, adapt this based on your use case
    
    # Evaluate the stories and save results
    evaluate_stories(
        grouped_dfs, 
        client, 
        "{}/{}".format(story_dir, args.run_id)
    )

if __name__ == "__main__":
    main()

