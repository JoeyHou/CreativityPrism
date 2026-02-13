import json
import jsonlines
from datetime import datetime
from pprint import pprint
import base64
import mimetypes
import os
from tqdm import tqdm 

def load_txt_prompt(filename):
    """
    Load a prompt from local txt file
    """
    prompt = ''.join(open(filename, 'r').readlines())
    return prompt

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

def write_json(data, filepath):
    # assert isinstance(data, dict), '[ERROR] Expect dictionary data!'
    json_data = data
    if isinstance(data, dict):
        json_data = {str(k): data[k] for k in data} # convert deys into str
    json_string = json.dumps(json_data, indent = 4)
    with open(filepath, 'w') as outfile:
        outfile.write(json_string)

def load_jsonl(filename):
    file_content = []
    try:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                file_content.append(obj)
            return file_content
    except FileNotFoundError:
        return []

def write_jsonl(data, filepath):
    with open(filepath, 'w') as jsonl_file:
        for line in data:
            jsonl_file.write(json.dumps(line))
            jsonl_file.write('\n')

def openai_call(messages, client, config):
    # set parameters
    temperature = config['temperature'] if 'temperature' in config else 0.75
    max_tokens = config['max_tokens'] if 'max_tokens' in config else 250
    # stop_tokens = config['stop_tokens'] if 'stop_tokens' in config else ['###']
    frequency_penalty = config['frequency_penalty'] if 'frequency_penalty' in config else 0
    presence_penalty = config['presence_penalty'] if 'presence_penalty' in config else 0
    # wait_time = config['wait_time']  if 'wait_time' in config else 0
    model = config['model'] if 'model' in config else 'text-dacinvi-003'
    return_logprobs = config['return_logprobs'] if 'return_logprobs' in config else False
    logprobs = True if return_logprobs else None
    response = client.chat.completions.create(
            model=model,
            messages = messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            # stop_tokens=stop_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs
    )
    completion = response.choices[0].message.content.strip() 
    if logprobs:
        logprobs = response.choices[0].logprobs
        return completion, logprobs
    else:
        return completion



def get_current_timestamp():
    return str(datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')[4:]



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
    if open_model:
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
                # {
                #     "temperature": config.get('temperature', 0),
                #     "frequency_penalty": config.get('frequency_penalty', 0),
                #     "presence_penalty": config.get('presence_penalty', 0),
                #     "max_tokens": config.get('max_tokens', 1024),
                # }
            )
            for d in tqdm(all_prompt_data)
        ]

    for i in range(len(all_prompt_data)):
        all_prompt_data[i]['raw_output'] = llm_outputs[i]
    return all_prompt_data