"""
Unified model interface for CreativityPrism.
Supports multiple model types: HuggingFace, OpenAI, Anthropic, vLLM, etc.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelInterface:
    """Unified interface for interacting with different LLM APIs."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = 'auto',
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize model interface.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of model API ('huggingface', 'openai', 'anthropic', 'vllm', 'auto')
            api_key: API key (if required)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.model_type = self._detect_model_type(model_name) if model_type == 'auto' else model_type
        self.api_key = api_key
        self.config = kwargs
        
        # Initialize the appropriate client
        self.client = self._initialize_client()
        logger.info(f"Initialized {self.model_type} model: {model_name}")
    
    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type from model name."""
        model_name_lower = model_name.lower()
        
        if any(x in model_name_lower for x in ['gpt-', 'davinci', 'curie']):
            return 'openai'
        elif any(x in model_name_lower for x in ['claude']):
            return 'anthropic'
        elif any(x in model_name_lower for x in ['gemini', 'palm']):
            return 'google'
        else:
            # Default to huggingface for local models
            return 'huggingface'
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.model_type == 'openai':
            return self._init_openai()
        elif self.model_type == 'anthropic':
            return self._init_anthropic()
        elif self.model_type == 'google':
            return self._init_google()
        elif self.model_type == 'huggingface':
            return self._init_huggingface()
        elif self.model_type == 'vllm':
            return self._init_vllm()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            if self.api_key:
                openai.api_key = self.api_key
            return openai
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def _init_google(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            return genai
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
    
    def _init_huggingface(self):
        """Initialize HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            return {'model': model, 'tokenizer': tokenizer}
        except ImportError:
            raise ImportError("Transformers package not installed. Run: pip install transformers torch")
    
    def _init_vllm(self):
        """Initialize vLLM client."""
        try:
            from vllm import LLM
            return LLM(model=self.model_name, **self.config)
        except ImportError:
            raise ImportError("vLLM package not installed. Run: pip install vllm")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.75,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model_type == 'openai':
            return self._generate_openai(prompt, temperature, max_tokens, top_p, **kwargs)
        elif self.model_type == 'anthropic':
            return self._generate_anthropic(prompt, temperature, max_tokens, top_p, **kwargs)
        elif self.model_type == 'google':
            return self._generate_google(prompt, temperature, max_tokens, top_p, **kwargs)
        elif self.model_type == 'huggingface':
            return self._generate_huggingface(prompt, temperature, max_tokens, top_p, **kwargs)
        elif self.model_type == 'vllm':
            return self._generate_vllm(prompt, temperature, max_tokens, top_p, **kwargs)
    
    def _generate_openai(self, prompt, temperature, max_tokens, top_p, **kwargs):
        """Generate using OpenAI API."""
        response = self.client.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt, temperature, max_tokens, top_p, **kwargs):
        """Generate using Anthropic API."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
    
    def _generate_google(self, prompt, temperature, max_tokens, top_p, **kwargs):
        """Generate using Google Generative AI."""
        model = self.client.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'top_p': top_p,
                **kwargs
            }
        )
        return response.text
    
    def _generate_huggingface(self, prompt, temperature, max_tokens, top_p, **kwargs):
        """Generate using HuggingFace model."""
        import torch
        
        tokenizer = self.client['tokenizer']
        model = self.client['model']
        
        inputs = tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def _generate_vllm(self, prompt, temperature, max_tokens, top_p, **kwargs):
        """Generate using vLLM."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self.client.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
