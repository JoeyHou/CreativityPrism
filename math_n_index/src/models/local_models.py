import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from configs import config

max_new_tokens = config["model_config"]["max_new_tokens"]
do_sample = config["model_config"]["do_sample"]
temperature = config["model_config"]["temperature"]
top_k = config["model_config"]["top_k"]
top_p = config["model_config"]["top_p"]


def load_local_model(model_name):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)

    if model_name in [
        "Deepseek-math-7b-rl",
        "Llama-3-70B",
        "Mixtral-8x22B",
        "Qwen1.5-72B",
    ]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif model_name == "Internlm2-math-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = model.eval()
    elif model_name == "Yi-1.5-34B":
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Local model {model_name} is not supported.")

    logger.info(f"Loading model {model_name}...")
    return model, tokenizer


def generate_local_response(model_name, model, tokenizer, messages):
    logger = logging.getLogger(__name__)
    model_id = config["model_version"].get(model_name)

    if model_name == "Deepseek-math-7b-rl":
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        model.generation_config = GenerationConfig.from_pretrained(model_id)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=None if temperature == 0 else temperature,
        )

        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    elif model_name == "Internlm2-math-20b":
        response, history = model.chat(
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=None if temperature == 0 else temperature,
            history=[],
            meta_instruction="",
        )
    elif model_name == "Mixtral-8x22B":
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=None if temperature == 0 else temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    elif model_name == "Llama-3-70B":
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=None if temperature == 0 else temperature,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    elif model_name == "Qwen1.5-72B":
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=None if temperature == 0 else temperature,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif model_name == "Yi-1.5-34B":
        input_ids = tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=None if temperature == 0 else temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
    else:
        logger.error(f"Response generation for {model_name} is not implemented.")
        raise ValueError(f"Response generation for {model_name} is not implemented.")

    return response.strip()  # Clean whitespaces
