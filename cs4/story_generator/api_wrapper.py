import openai
import anthropic

from google import genai
from google.genai import types

# import google.generativeai as genai
# from google.generativeai import types

class ModelWrapper:
    """
    A wrapper to handle different API-based LLM models.
    Supports OpenAI (GPT), Anthropic (Claude), and Google Gemini.
    Can extends to more if needed.
    """
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        
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
            self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self.provider = "deepseek"
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
    
    def generate_response(self, messages, config = {}):
        """
        Routes the request to the correct API endpoint based on the selected model.
        """
        if self.provider == "openai":
            return self._query_openai(messages, config)
        elif self.provider == "anthropic":
            return self._query_anthropic(messages, config)
        elif self.provider == "gemini":
            return self._query_gemini(messages, config)
        elif self.provider == "deepseek":
            return self._query_deepseek(messages, config)
    
    def _query_openai(self, messages, config):
        """
        Queries OpenAI's chat model.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages, #[{"role": "user", "content": prompt}],
                max_tokens=config.get('max_tokens', 128),
                temperature=config.get('temperature', 0.75)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"
    
    def _query_anthropic(self, messages, config):
        """
        Queries Anthropic's Claude model using the latest SDK.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages, #[{"role": "user", "content": prompt}],
                max_tokens=config.get('max_tokens', 1024),
                temperature=config.get('temperature', 0.75)
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API Error: {e}"
    
    def _query_gemini(self, messages, config):
        """
        Queries Google's Gemini model using the genai client.
        """
        try:
            # print(prompt)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="\n".join([
                    "[{}]: {}".format(mes['role'], mes['content'])
                    for mes in messages
                ]), # [{"role": "user", "content": prompt}] => [{role}]: {content}
                config=types.GenerateContentConfig(
                    max_output_tokens=config.get('max_tokens', 1024),
                    temperature=config.get('temperature', 0.75),
                    seed=42
                )
            )
            # print(response)
            return response.text
        except Exception as e:
            return f"Gemini API Error: {e}"
    
    def _query_deepseek(self, messages, config):
        """
        Queries DeepSeek model using the openai client.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=config.get('max_tokens', 1024),
                temperature=config.get('temperature', 0.75),
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"DeepSeek API Error: {e}"