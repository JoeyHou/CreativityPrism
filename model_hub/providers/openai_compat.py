"""OpenAI wire format: OpenAI, DeepSeek, vLLM's server, and any compatible endpoint.

This is the one that makes a custom `base_url` possible. The previous wrappers
hardcoded DeepSeek's URL as the only non-default endpoint and had no way to
reach anything else -- no self-hosted vLLM, no gateway, no Together/Fireworks.
"""


class OpenAICompatProvider:
    def __init__(self, model):
        import openai
        self.model = model
        kwargs = {"api_key": model.api_key}
        if model.base_url:
            kwargs["base_url"] = model.base_url
        self.client = openai.OpenAI(**kwargs)

    def generate(self, messages, max_tokens=None, temperature=None, top_p=None,
                 top_k=None, seed=None, **_):
        # top_k has no place in the OpenAI schema; dropped rather than passed
        # through, which would 400 on a strict endpoint.
        kwargs = {
            "model": self.model.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None:
            kwargs["seed"] = seed
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def complete(self, prompt, max_tokens=None, temperature=None, top_p=None, **_):
        """Raw completion, no chat template -- the apples-to-apples equivalent of
        vLLM's offline `LLM.generate(text)`. Used against a vLLM server."""
        resp = self.client.completions.create(
            model=self.model.model_id, prompt=prompt,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        return resp.choices[0].text
