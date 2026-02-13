# from vllm.utils import get_openai_chat_template

from configs import config

USE_OPENAI_TEMPLATE = config["prompt"]["use_openai_template"]

def load_messages(model_name, prompt):
    # we can always add more models
    templates = {
        # Models via API calls
        "claude-3-opus": [
            {"role": "user", "content": prompt},
        ],
        "claude-3-5-sonnet": [
            {"role": "user", "content": prompt},
        ],
        "deepseek-v2": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "gemini-1.5-pro": prompt,  # Gemini uses prompt (string) instead of message list.
        "gpt-4": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gpt-4o": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gpt-4o-mini": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # Models run locally
        "Deepseek-math-7b-rl": [
            {"role": "user", "content": prompt},
        ],
        "Internlm2-math-20b": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "Llama-3.1-70B": [
            {"role": "user", "content": prompt},
        ],
        "Llama-3.3-70B": [
            {"role": "user", "content": prompt},
        ],
        "Llama-3.1-8B": [
            {"role": "user", "content": prompt},
        ],
        "Mixtral-8x22B": [{"role": "user", "content": prompt}],
        "Qwen1.5-72B": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "Yi-1.5-34B": [{"role": "user", "content": prompt}],
    }
    return templates.get(model_name)


def load_messages_vllm(model_name, prompt, use_openai_format=USE_OPENAI_TEMPLATE): # default to true
    """
    Returns a formatted prompt for vLLM generation.
    If `use_openai_format` is True, uses OpenAI chat templates.
    """
    messages = load_messages(model_name, prompt)

    # Ensure messages is always a list of dictionaries
    # We first test this part for toy
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        return messages
    elif isinstance(messages, str):  # Convert raw strings into a chat format
        messages = [{"role": "user", "content": messages}]
        return messages
    else:
        return messages # incorrect format and is not None

    # Use OpenAI format only for models that expect it
    openai_like_models = [
        "gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-opus", 
        "claude-3-5-sonnet", "deepseek-v2", "Qwen1.5-72B"
    ]
    if use_openai_format and model_name in openai_like_models:
        # return get_openai_chat_template(messages) 
        return "Not setup yet"

    # Otherwise, manually format the prompt
    # TODO: please check if we need these
    templates = {
        # API-based models (Unchanged)
        "claude-3-opus": f"User: {prompt}\nAssistant:",
        "claude-3-5-sonnet": f"User: {prompt}\nAssistant:",
        "deepseek-v2": f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:",
        "gemini-1.5-pro": prompt,  # Gemini uses raw prompt
        "gpt-4": f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:",
        "gpt-4o": f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:",
        "gpt-4o-mini": f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:",
        
        # Local vLLM models (Ensure correct format)
        "Deepseek-math-7b-rl": f"User: {prompt}\nAssistant:",
        "Internlm2-math-20b": f"System: \nUser: {prompt}\nAssistant:",
        "Llama-3-70B": f"System: \nUser: {prompt}\nAssistant:",
        "Mixtral-8x22B": f"User: {prompt}\nAssistant:",
        "Qwen1.5-72B": f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:",
        "Yi-1.5-34B": f"User: {prompt}\nAssistant:",
    }

    return templates.get(model_name, f"User: {prompt}\nAssistant:")  # Default manual format

