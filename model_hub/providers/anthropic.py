"""Anthropic messages API."""


class AnthropicProvider:
    def __init__(self, model):
        import anthropic
        self.model = model
        kwargs = {"api_key": model.api_key}
        if model.base_url:
            kwargs["base_url"] = model.base_url
        self.client = anthropic.Anthropic(**kwargs)

    def generate(self, messages, max_tokens=None, temperature=None, top_p=None,
                 top_k=None, seed=None, **_):
        # Anthropic has no seed parameter; silently dropped.
        system = None
        convo = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content")
            else:
                convo.append(m)
        kwargs = {
            "model": self.model.model_id,
            "messages": convo,
            "max_tokens": max_tokens or 1024,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text
