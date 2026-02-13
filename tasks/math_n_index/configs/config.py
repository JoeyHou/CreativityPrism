import json


def load_config():
    file_path = "/scratch/dkhasha1/bzhang90/creative_bench/configs/config.json"
    with open(file_path, "r") as file:
        return json.load(file)


config = load_config()