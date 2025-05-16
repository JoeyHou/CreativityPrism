import openai
import anthropic
from google import genai
from google.genai import types
from openai import OpenAI

class ModelWrapper:
    """
    A wrapper to handle different API-based LLM models.
    Supports OpenAI (GPT), Anthropic (Claude), and Google Gemini.
    Can extends to more if needed.
    """
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

        self.max_tokens = 512
        self.temperature = 1
        self.top_p = 1
        self.top_k = 50
        
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
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
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
                temperature=self.temperature, 
                top_p=self.top_p,
                top_k=self.top_k
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API Error: {e}"
        
    def _query_gemini(self, prompt):
        """
        Queries Google's Gemini model using the genai client.
        """
        try:
            # print(prompt)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        seed=42
                )
            )
            return response.text
        except Exception as e:
            return f"Gemini API Error: {e}"
        
    def _query_deepseek(self, prompt):
        """
        Queries DeepSeek model using the openai client.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"DeepSeek API Error: {e}"