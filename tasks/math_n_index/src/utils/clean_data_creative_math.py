"""Strip novelty commentary from creative_math solutions.

Runs BETWEEN inference and evaluation: it reads the `response` field written by
run_inference_all.py and adds the `cleaned_response` field that
creative_math_eval_api.py scores. Evaluation raises KeyError without it.

The cleaner is a fixed instrument, never the model under test: every model in a
comparison must be cleaned by the same one, or their scores are not comparable.
The published numbers use the vllm backend with Llama-3.3-70B. The openai backend
exists so this step can run without a GPU, but it produces different cleaned text.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

TEMPERATURE = 0.1
MAX_TOKENS = 3000
SEED = 14


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


def generate_vllm(prompts, model_name, tensor_parallel_size):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED)
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def generate_openai(prompts, model_name, max_workers):
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("clean_data_creative_math.py: OPENAI_API_KEY is not set.")

    client = OpenAI(max_retries=5)

    def clean_one(prompt):
        completion = client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            messages=[{"role": "user", "content": prompt}],
        )
        return (completion.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(clean_one, prompts))


def main():
    parser = argparse.ArgumentParser(description="Clean solution responses before creative_math evaluation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the cleaned JSON file.")
    parser.add_argument("--backend", choices=["vllm", "openai"], default="vllm",
                        help="vllm reproduces the published setup; openai runs without a GPU.")
    parser.add_argument("--model_name", type=str, default=None,
                        help=f"The cleaner, NOT the model under test. Defaults to {DEFAULT_VLLM_MODEL} "
                             f"for vllm, {DEFAULT_OPENAI_MODEL} for openai.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="vllm backend only.")
    parser.add_argument("--max_workers", type=int, default=8, help="openai backend only.")
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = DEFAULT_VLLM_MODEL if args.backend == "vllm" else DEFAULT_OPENAI_MODEL

    with open(args.input_file) as f:
        data = json.load(f)

    prompts = [build_prompt(item["response"]) for item in data]

    if args.backend == "vllm":
        raw_outputs = generate_vllm(prompts, args.model_name, args.tensor_parallel_size)
    else:
        raw_outputs = generate_openai(prompts, args.model_name, args.max_workers)

    # Regular expression pattern to extract between START and END
    pattern = re.compile(r"START\s*(.*?)\s*END", re.DOTALL)

    for item, raw_output in zip(data, raw_outputs):
        item["raw_cleaned_response"] = raw_output

        match = pattern.search(raw_output)
        if match:
            item["cleaned_response"] = match.group(1).strip()
            item["extraction_status"] = "extracted"
        else:
            item["cleaned_response"] = item["response"].strip()  # fallback to original response
            item["extraction_status"] = "fallback"

        # Lets a reader tell whether these scores are comparable to the published run.
        item["cleaner_model"] = args.model_name

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)

    fallbacks = sum(1 for item in data if item["extraction_status"] == "fallback")
    print(f"Cleaned {len(data)} item(s) with {args.model_name}; {fallbacks} fell back to the raw response.")


if __name__ == "__main__":
    main()
