import json
import logging
import os
import re


def load_json(data_path):
    logger = logging.getLogger(__name__)
    with open(data_path, "r") as file:
        data = json.load(file)
    logger.info(f"JSON data loaded from {os.path.abspath(data_path)}")
    return data


def save_json(data, data_path):
    logger = logging.getLogger(__name__)
    with open(data_path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(data_path)}")


def extract_yes_no(response):
    # Whole words only, first verdict wins: "YES" in response also fired on "YESTERDAY", and
    # a lowercase "yes" used to score as NO. Prompts put the verdict before the explanation.
    match = re.search(r"\b(YES|NO)\b", response, re.IGNORECASE)
    return match.group(1).upper() if match else "NO"
