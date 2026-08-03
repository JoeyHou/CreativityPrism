import json
import os
from pathlib import Path


def load_config():
    file_path = os.environ.get(
        "CREATIVITYPRISM_MATH_CONFIG", Path(__file__).with_name("config.json")
    )
    with open(file_path, "r") as file:
        return json.load(file)


config = load_config()