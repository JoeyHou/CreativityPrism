"""
CreativityPrism Unified Generation Script
==========================================
A unified script to generate responses for all tasks in the CreativityPrism benchmark.

Tasks covered:
1. Divergent Thinking Domain:
   - AUT (Alternative Uses Task)
   - DAT (Divergent Association Task)
   - TTCT (Torrance Test of Creative Thinking)

2. Creative Writing Domain:
   - TTCW (Torrance Test of Creative Writing)
   - Creative Short Story
   - Creativity Index
   - CS4 (Creative Story with 4 constraints)

3. Logical Reasoning Domain:
   - NeoCoder
   - Creative Math
"""

import argparse
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Enum for all CreativityPrism tasks"""
    # Divergent Thinking
    AUT = "aut"  # Alternative Uses Task
    DAT = "dat"  # Divergent Association Task
    TTCT = "ttct"  # Torrance Test of Creative Thinking
    
    # Creative Writing
    TTCW = "ttcw"  # Torrance Test of Creative Writing
    CREATIVE_SHORT_STORY = "creative_short_story"
    CREATIVITY_INDEX = "creativity_index"
    CS4 = "cs4"  # Creative Story with 4 constraints
    
    # Logical Reasoning
    NEOCODER = "neocoder"
    CREATIVE_MATH = "creative_math"


class DomainType(Enum):
    """Enum for the three domains"""
    DIVERGENT_THINKING = "divergent_thinking"
    CREATIVE_WRITING = "creative_writing"
    LOGICAL_REASONING = "logical_reasoning"


@dataclass
class GenerationConfig:
    """Configuration for generation parameters"""
    temperature: float = 0.75
    max_tokens: int = 4096
    top_p: float = 1.0
    num_samples: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "num_samples": self.num_samples
        }


class TaskPromptBuilder:
    """Builds prompts for each task type"""
    
    @staticmethod
    def build_aut_prompt(tool_name: str) -> str:
        """Alternative Uses Task: Generate creative alternative uses for a tool"""
        return f"""Create a list of creative alternative uses for a {tool_name}.

Think of unconventional and novel ways to use this item beyond its typical purpose. Be creative and imaginative.

Please provide your response as a numbered list of alternative uses."""
    
    @staticmethod
    def build_dat_prompt() -> str:
        """Divergent Association Task: Generate semantically distant nouns"""
        return """Please write 10 nouns in English that are as irrelevant from each other as possible, in all meanings and uses of the words.

The words should be as semantically distant as possible. Provide them as a comma-separated list."""
    
    @staticmethod
    def build_ttct_prompt(question: str) -> str:
        """Torrance Test of Creative Thinking"""
        return f"""Question: {question}

Please provide a creative and detailed response to this question. Think divergently and consider multiple perspectives, unusual ideas, and elaborate on your thoughts."""
    
    @staticmethod
    def build_ttcw_prompt(plot: str) -> str:
        """Torrance Test of Creative Writing: Write story from plot"""
        return f"""Write a New Yorker-style story given the plot below. Make sure it is at least 2000 words.

Plot: {plot}

Story:"""
    
    @staticmethod
    def build_creative_short_story_prompt(keywords: List[str]) -> str:
        """Creative Short Story: Use keywords in unconventional ways"""
        keyword_str = ", ".join(keywords)
        return f"""You need to come up with a novel and unique story that uses the required words in unconventional ways or settings. Make sure you use at most five sentences.

The given words: {keyword_str}

Story:"""
    
    @staticmethod
    def build_creativity_index_prompt(prefix: str) -> str:
        """Creativity Index: Continue text from prefix"""
        return f"""{prefix}"""
    
    @staticmethod
    def build_cs4_prompt(base_story: str, constraints: List[str]) -> str:
        """CS4: Write story with constraints"""
        constraints_str = "\n".join([f"- {c}" for c in constraints])
        return f"""BaseStory: "{base_story}"

Now modify the existing story to accommodate the following constraints:
{constraints_str}

Come up with a new story in 500 words.

Story:"""
    
    @staticmethod
    def build_neocoder_prompt(problem: str, constraints: List[str] = None) -> str:
        """NeoCoder: Solve coding problem with constraints"""
        prompt = f"""Problem: {problem}

"""
        if constraints:
            constraints_str = "\n".join([f"- {c}" for c in constraints])
            prompt += f"""Constraints:
{constraints_str}

"""
        prompt += "Please provide your solution:"
        return prompt
    
    @staticmethod
    def build_creative_math_prompt(question: str, reference_solutions: List[str] = None) -> str:
        """Creative Math: Solve math problem creatively"""
        prompt = f"""Question: {question}

"""
        if reference_solutions:
            for i, sol in enumerate(reference_solutions, 1):
                prompt += f"Reference Solution {i}: {sol}\n\n"
            prompt += "Please provide a DIFFERENT solution approach:\n"
        else:
            prompt += "Please provide your solution:\n"
        return prompt


class TaskConfigManager:
    """Manages task-specific configurations"""
    
    TASK_CONFIGS = {
        TaskType.AUT: {
            "domain": DomainType.DIVERGENT_THINKING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=1024),
            "num_rounds": 5,  # Generate 5 responses per tool
        },
        TaskType.DAT: {
            "domain": DomainType.DIVERGENT_THINKING,
            "generation_config": GenerationConfig(temperature=1.0, max_tokens=256),
            "num_rounds": 100,  # Generate 100 rounds
        },
        TaskType.TTCT: {
            "domain": DomainType.DIVERGENT_THINKING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=2048),
        },
        TaskType.TTCW: {
            "domain": DomainType.CREATIVE_WRITING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=4096),
        },
        TaskType.CREATIVE_SHORT_STORY: {
            "domain": DomainType.CREATIVE_WRITING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=512),
        },
        TaskType.CREATIVITY_INDEX: {
            "domain": DomainType.CREATIVE_WRITING,
            "generation_config": GenerationConfig(temperature=1.0, max_tokens=288, top_p=0.9),
        },
        TaskType.CS4: {
            "domain": DomainType.CREATIVE_WRITING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=1024),
            "num_samples_per_constraint": 3,  # For diversity metrics
        },
        TaskType.NEOCODER: {
            "domain": DomainType.LOGICAL_REASONING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=2048),
        },
        TaskType.CREATIVE_MATH: {
            "domain": DomainType.LOGICAL_REASONING,
            "generation_config": GenerationConfig(temperature=0.75, max_tokens=1536),
        },
    }
    
    @classmethod
    def get_config(cls, task: TaskType) -> Dict[str, Any]:
        """Get configuration for a specific task"""
        return cls.TASK_CONFIGS.get(task, {})
    
    @classmethod
    def get_generation_config(cls, task: TaskType) -> GenerationConfig:
        """Get generation config for a task"""
        config_dict = cls.get_config(task)
        return config_dict.get("generation_config", GenerationConfig())


class CreativityPrismGenerator:
    """Main generator class for CreativityPrism tasks"""
    
    def __init__(self, model_client, model_name: str):
        """
        Initialize generator with a model client
        
        Args:
            model_client: The LLM client (e.g., OpenAI, Anthropic, vLLM client)
            model_name: Name of the model being used
        """
        self.model_client = model_client
        self.model_name = model_name
        self.prompt_builder = TaskPromptBuilder()
        self.config_manager = TaskConfigManager()
    
    def generate(
        self,
        task: TaskType,
        input_data: Dict[str, Any],
        output_path: Optional[str] = None,
        custom_config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate responses for a given task
        
        Args:
            task: The task type to generate for
            input_data: Input data specific to the task
            output_path: Optional path to save outputs
            custom_config: Optional custom generation configuration
            
        Returns:
            Dictionary containing generated outputs and metadata
        """
        # Get task configuration
        config = custom_config or self.config_manager.get_generation_config(task)
        
        # Build prompt based on task type
        prompt = self._build_prompt(task, input_data)
        
        # Generate response
        response = self._call_model(prompt, config)
        
        # Package result
        result = {
            "task": task.value,
            "model": self.model_name,
            "input_data": input_data,
            "prompt": prompt,
            "generation_config": config.to_dict(),
            "output": response,
        }
        
        # Save if output path provided
        if output_path:
            self._save_result(result, output_path)
        
        return result
    
    def batch_generate(
        self,
        task: TaskType,
        input_data_list: List[Dict[str, Any]],
        output_dir: str,
        custom_config: Optional[GenerationConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple inputs
        
        Args:
            task: The task type
            input_data_list: List of input data dictionaries
            output_dir: Directory to save outputs
            custom_config: Optional custom generation configuration
            
        Returns:
            List of result dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, input_data in enumerate(input_data_list):
            print(f"Generating for {task.value} - Sample {i+1}/{len(input_data_list)}")
            
            output_path = os.path.join(output_dir, f"{task.value}_{i:04d}.json")
            result = self.generate(task, input_data, output_path, custom_config)
            results.append(result)
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{task.value}_summary.json")
        self._save_summary(results, summary_path)
        
        return results
    
    def _build_prompt(self, task: TaskType, input_data: Dict[str, Any]) -> str:
        """Build prompt for specific task"""
        if task == TaskType.AUT:
            return self.prompt_builder.build_aut_prompt(input_data["tool_name"])
        
        elif task == TaskType.DAT:
            return self.prompt_builder.build_dat_prompt()
        
        elif task == TaskType.TTCT:
            return self.prompt_builder.build_ttct_prompt(input_data["question"])
        
        elif task == TaskType.TTCW:
            return self.prompt_builder.build_ttcw_prompt(input_data["plot"])
        
        elif task == TaskType.CREATIVE_SHORT_STORY:
            return self.prompt_builder.build_creative_short_story_prompt(
                input_data["keywords"]
            )
        
        elif task == TaskType.CREATIVITY_INDEX:
            return self.prompt_builder.build_creativity_index_prompt(
                input_data["prefix"]
            )
        
        elif task == TaskType.CS4:
            return self.prompt_builder.build_cs4_prompt(
                input_data["base_story"],
                input_data["constraints"]
            )
        
        elif task == TaskType.NEOCODER:
            return self.prompt_builder.build_neocoder_prompt(
                input_data["problem"],
                input_data.get("constraints")
            )
        
        elif task == TaskType.CREATIVE_MATH:
            return self.prompt_builder.build_creative_math_prompt(
                input_data["question"],
                input_data.get("reference_solutions")
            )
        
        else:
            raise ValueError(f"Unknown task type: {task}")
    
    def _call_model(self, prompt: str, config: GenerationConfig) -> str:
        """
        Call the model with the given prompt and configuration
        
        This is a placeholder that should be implemented based on your model client
        """
        # This is where you'd implement the actual model call
        # For different APIs (OpenAI, Anthropic, vLLM, etc.)
        raise NotImplementedError(
            "Model calling logic needs to be implemented based on your specific model client"
        )
    
    def _save_result(self, result: Dict[str, Any], output_path: str):
        """Save a single result to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self, results: List[Dict[str, Any]], summary_path: str):
        """Save summary of batch generation"""
        summary = {
            "model": self.model_name,
            "task": results[0]["task"] if results else None,
            "num_samples": len(results),
            "generation_config": results[0]["generation_config"] if results else None,
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def load_input_data(task: TaskType, data_file: str) -> List[Dict[str, Any]]:
    """
    Load input data for a specific task
    
    Args:
        task: The task type
        data_file: Path to the input data file
        
    Returns:
        List of input data dictionaries
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format data based on task requirements
    formatted_data = []
    
    if task == TaskType.AUT:
        # Expect: list of tool names
        for item in data:
            formatted_data.append({"tool_name": item})
    
    elif task == TaskType.DAT:
        # No input needed, just generate N rounds
        formatted_data = [{}] * data.get("num_rounds", 100)
    
    elif task == TaskType.TTCT:
        # Expect: list of questions
        for item in data:
            formatted_data.append({"question": item})
    
    elif task == TaskType.TTCW:
        # Expect: list of plots
        for item in data:
            formatted_data.append({"plot": item})
    
    elif task == TaskType.CREATIVE_SHORT_STORY:
        # Expect: list of keyword tuples
        for item in data:
            formatted_data.append({"keywords": item})
    
    elif task == TaskType.CREATIVITY_INDEX:
        # Expect: list of prefixes
        for item in data:
            formatted_data.append({"prefix": item})
    
    elif task == TaskType.CS4:
        # Expect: base stories with constraints
        for item in data:
            formatted_data.append({
                "base_story": item["base_story"],
                "constraints": item["constraints"]
            })
    
    elif task == TaskType.NEOCODER:
        # Expect: coding problems
        for item in data:
            formatted_data.append({
                "problem": item["problem"],
                "constraints": item.get("constraints")
            })
    
    elif task == TaskType.CREATIVE_MATH:
        # Expect: math questions with optional reference solutions
        for item in data:
            formatted_data.append({
                "question": item["question"],
                "reference_solutions": item.get("reference_solutions")
            })
    
    return formatted_data


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="CreativityPrism Unified Generation Script"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[t.value for t in TaskType],
        help="Task to generate for"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file (JSON format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated outputs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override default temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Override default max tokens"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Override default top_p"
    )
    
    args = parser.parse_args()
    
    # Convert task string to enum
    task = TaskType(args.task)
    
    # Load input data
    print(f"Loading input data from {args.input}")
    input_data_list = load_input_data(task, args.input)
    print(f"Loaded {len(input_data_list)} samples")
    
    # Create custom config if overrides provided
    custom_config = None
    if any([args.temperature, args.max_tokens, args.top_p]):
        default_config = TaskConfigManager.get_generation_config(task)
        custom_config = GenerationConfig(
            temperature=args.temperature or default_config.temperature,
            max_tokens=args.max_tokens or default_config.max_tokens,
            top_p=args.top_p or default_config.top_p,
        )
    
    # Initialize generator
    # NOTE: You need to implement your model client here
    # Example for different APIs:
    # - OpenAI: model_client = OpenAI(api_key=...)
    # - Anthropic: model_client = Anthropic(api_key=...)
    # - vLLM: model_client = LLM(model=...)
    
    print("Initializing model client...")
    model_client = None  # TODO: Initialize your model client
    
    generator = CreativityPrismGenerator(model_client, args.model_name)
    
    # Generate
    print(f"Generating for task: {task.value}")
    results = generator.batch_generate(
        task=task,
        input_data_list=input_data_list,
        output_dir=args.output_dir,
        custom_config=custom_config
    )
    
    print(f"Generation complete! Results saved to {args.output_dir}")
    print(f"Total samples generated: {len(results)}")


if __name__ == "__main__":
    main()