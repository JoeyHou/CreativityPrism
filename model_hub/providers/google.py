"""Google Gemini.

Carries the thinking-budget fix that previously existed in only one of the four
wrappers. Gemini 2.5+ spends the output budget on thinking first, so a response
under a small max_output_tokens comes back with `finishReason=MAX_TOKENS` and
**no `parts` key at all** -- the wrapper then raises KeyError or returns empty
text, and a judge silently scores every item as a failure. Setting
thinking_budget=0 is what makes a short verdict come back at all.
"""


class GoogleProvider:
    def __init__(self, model):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=model.api_key)

    def generate(self, messages, max_tokens=None, temperature=None, top_p=None,
                 top_k=None, seed=None, thinking_budget=0, **_):
        from google.genai import types
        # Flattened the way the aut wrapper did it, so a multi-turn prompt keeps
        # its role markers rather than being silently concatenated.
        if len(messages) == 1 and messages[0].get("role") == "user":
            contents = messages[0]["content"]
        else:
            contents = "\n".join(
                "[{}]: {}".format(m.get("role", "user"), m.get("content", ""))
                for m in messages
            )
        cfg = {"max_output_tokens": max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        if top_p is not None:
            cfg["top_p"] = top_p
        if top_k is not None:
            cfg["top_k"] = top_k
        if seed is not None:
            cfg["seed"] = seed
        if thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        resp = self.client.models.generate_content(
            model=self.model.model_id, contents=contents,
            config=types.GenerateContentConfig(**cfg),
        )
        return resp.text
