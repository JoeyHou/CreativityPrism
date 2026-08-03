import openai
import anthropic
from google import genai
from google.genai import types
from openai import OpenAI
import json
import os
import time

# Applies to callers that do not pass max_tokens explicitly -- in practice the judges.
# Every judge prompt asks for a verdict *and* an explanation, and most current models write
# the explanation first. At the original hardcoded 30 tokens the reply is cut off before the
# verdict appears, extract_yes_no() sees no YES/NO, and the sample scores as UNCLEAR/NO --
# which the unanimity rule turns into a 0% correctness ratio for the whole run.
# Set CREATIVITYPRISM_MATH_API_MAX_TOKENS=30 to reproduce the original truncated behavior.
DEFAULT_MAX_TOKENS = int(os.environ.get("CREATIVITYPRISM_MATH_API_MAX_TOKENS", "512"))

class ModelWrapper:
    """
    A wrapper to handle different API-based LLM models.
    Supports OpenAI (GPT), Anthropic (Claude), and Google Gemini.
    Can extends to more if needed.
    """
    def __init__(self, model_name, api_key, max_tokens=None):
        self.model_name = model_name
        self.api_key = api_key
        # Generation callers pass the configured max_new_tokens; judges leave it unset.
        self.max_tokens = int(max_tokens) if max_tokens else DEFAULT_MAX_TOKENS
        
        if "gpt" in self.model_name.lower():
            self.client = openai.OpenAI(api_key=api_key)
            self.provider = "openai"
        elif "claude" in self.model_name.lower():
            self.client = anthropic.Anthropic(api_key=api_key)
            self.provider = "anthropic"
        elif "gemini" in self.model_name.lower():
            self.client = genai.Client(api_key=api_key)
            self.provider = "gemini"
        elif "deepseek" in self.model_name.lower():
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self.provider = "deepseek"
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
    
    def generate_response(self, prompt):
        """
        Routes the request to the correct API endpoint based on the selected model.
        """
        if self.provider == "openai":
            return self._query_openai(prompt)
        elif self.provider == "anthropic":
            return self._query_anthropic(prompt)
        elif self.provider == "gemini":
            return self._query_gemini(prompt)
        elif self.provider == "deepseek":
            return self._query_deepseek(prompt)
    
    def _query_openai(self, prompt):
        """
        Queries OpenAI's chat model.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"
    
    def _query_anthropic(self, prompt):
        """
        Queries Anthropic's Claude model using the latest SDK.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0
                # no seed can be added
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API Error: {e}"
    # remember to change back for differenit configs
    def _query_gemini(self, prompt, max_retries=3, retry_delay=0.5):
        """
        Queries Google's Gemini model using the genai client.
        Retries the request up to `max_retries` times on failure.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=0,
                        seed=42,
                        # Gemini 2.5+ spends the output budget on thinking first, so a
                        # 30-token verdict comes back with no text at all unless it is off.
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"Gemini API Error after {max_retries} attempts: {e}"
        
    def _query_deepseek(self, prompt):
        """
        Queries DeepSeek model using the openai client.
        If the model is deepseek-reasoner, logs reasoning_content to math_reasoner.jsonl.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.0,
                seed=42
            )

            if self.model_name == "deepseek-reasoner":
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
                # Save reasoning content to JSONL file
                if reasoning_content:
                    with open("math_reasoner_logs.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "prompt": prompt,
                            "reasoning_content": reasoning_content
                        }) + "\n")
            
            return response.choices[0].message.content

        except Exception as e:
            return f"DeepSeek API Error: {e}"
