import json
import re
import argparse
from vllm import LLM, SamplingParams

# Argument parser setup
parser = argparse.ArgumentParser(description="Clean solution responses using vLLM.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the cleaned JSON file.")
parser.add_argument("--model_name", type=str, default="qwen-72b-instruct", help="Name of the model to load.")
args = parser.parse_args()

# Load dataset
with open(args.input_file) as f:
    data = json.load(f)

# Load the vLLM model
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4, gpu_memory_utilization=0.9)

# Prompt builder
def build_prompt(original_solution):
    return (
        "You are given a solution to a math problem. Remove any sentence or clause that discusses the solution's "
        "novelty, uniqueness, or how it differs from other approaches. KEEP all content that explains the mathematical "
        "correctness and the process by which the final answer is derived. Do NOT paraphrase, re-order, or rewrite anything; "
        "simply delete novelty-related commentary and leave the rest unchanged. "
        "Do NOT add any additional comments. Do NOT extend the solution. "
        "ONLY output the cleaned solution enclosed between START and END tokens. "
        "Start your output with 'START' on its own line and end with 'END' on its own line."
        "\n\nOriginal solution:\n"
        f"{original_solution}\n\nCleaned solution:"
    )

# Build prompts
prompts = [build_prompt(item["response"]) for item in data]

# Sampling settings
sampling_params = SamplingParams(temperature=0.1, max_tokens=3000, seed=14)

# Batched inference
outputs = llm.generate(prompts, sampling_params)

# Regular expression pattern to extract between START and END
pattern = re.compile(r"START\s*(.*?)\s*END", re.DOTALL)

# Add cleaned responses
for item, output in zip(data, outputs):
    raw_output = output.outputs[0].text.strip()
    item["raw_cleaned_response"] = raw_output

    # Extract between START and END
    match = pattern.search(raw_output)
    if match:
        cleaned = match.group(1).strip()
        item["extraction_status"] = "extracted"
    else:
        cleaned = item["response"].strip()  # fallback to original response
        item["extraction_status"] = "fallback"

    item["cleaned_response"] = cleaned

# Save output
with open(args.output_file, "w") as f:
    json.dump(data, f, indent=2)