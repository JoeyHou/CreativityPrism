import json

def transform_data(input_data, dataset_name="creativity_index", prefix="speech"):
    transformed_data = []
    
    for idx, entry in enumerate(input_data):
        transformed_entry = {
            "meta_data": {
                "dataset": dataset_name,
                "id": f"{prefix}_{idx}"
            },
            "input": {
                "text": entry["prompt"]
            },
            "output": None
        }
        transformed_data.append(transformed_entry)

    return transformed_data

input_data_path = "/creative_bench/data/raw/creative_index_data/speech/ChatGPT_speech.json"

def load_json(data_path):
    with open(data_path, "r") as file:
        data = json.load(file)
    return data

input_data = load_json(input_data_path)

output_data = transform_data(input_data=input_data)

with open("speech_prompt.json", "w") as f:
    json.dump(output_data, f, indent=4)

