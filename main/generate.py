#!/usr/bin/env python3
"""
Unified generation script for CreativityPrism benchmark.
Executes any task with any model configuration.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

from tasks import TASK_REGISTRY
from models.model_interface import ModelInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Unified pipeline for generating responses across all CreativityPrism tasks."""
    
    def __init__(self, config_path: str = "config/tasks.yaml"):
        """Initialize the generation pipeline with task configurations."""
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load task configurations from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate(
        self,
        task_name: str,
        model_name: str,
        model_config: Dict[str, Any],
        output_dir: str = "outputs/generations",
        num_samples: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses for a specific task using a specific model.
        
        Args:
            task_name: Name of the task (e.g., 'aut', 'dat', 'ttcw')
            model_name: Name or path of the model
            model_config: Model configuration (temperature, max_tokens, etc.)
            output_dir: Directory to save generated outputs
            num_samples: Number of samples to generate (overrides config)
            **kwargs: Additional task-specific arguments
            
        Returns:
            Dictionary containing generated outputs and metadata
        """
        # Initialize task
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Task '{task_name}' not found. Available: {list(TASK_REGISTRY.keys())}")
        
        task_class = TASK_REGISTRY[task_name]
        task_config = self.config['tasks'].get(task_name, {})
        task_config.update(kwargs)
        
        task = task_class(**task_config)
        logger.info(f"Initialized task: {task_name}")
        
        # Initialize model
        model = ModelInterface(model_name, **model_config)
        logger.info(f"Initialized model: {model_name}")
        
        # Load dataset
        dataset = task.load_dataset()
        if num_samples:
            dataset = dataset[:num_samples]
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Generate responses
        results = []
        for idx, example in enumerate(dataset):
            logger.info(f"Generating {idx+1}/{len(dataset)}")
            
            # Get prompt for this example
            prompt = task.format_prompt(example)
            
            # Generate response
            response = model.generate(
                prompt,
                temperature=model_config.get('temperature', 0.75),
                max_tokens=model_config.get('max_tokens', 2048),
                top_p=model_config.get('top_p', 1.0)
            )
            
            # Store result
            result = {
                'task': task_name,
                'example_id': example.get('id', idx),
                'input': example,
                'prompt': prompt,
                'response': response,
                'metadata': {
                    'model': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'config': model_config
                }
            }
            results.append(result)
        
        # Save outputs
        output_path = Path(output_dir) / task_name / model_name.replace('/', '_')
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"generations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Saved {len(results)} generations to {output_file}")
        
        return {
            'task': task_name,
            'model': model_name,
            'num_generated': len(results),
            'output_file': str(output_file),
            'results': results
        }
    
    def generate_batch(
        self,
        tasks: List[str],
        model_name: str,
        model_config: Dict[str, Any],
        output_dir: str = "outputs/generations"
    ) -> Dict[str, Any]:
        """
        Generate responses for multiple tasks with a single model.
        
        Args:
            tasks: List of task names
            model_name: Name or path of the model
            model_config: Model configuration
            output_dir: Directory to save generated outputs
            
        Returns:
            Dictionary mapping task names to their results
        """
        all_results = {}
        
        for task_name in tasks:
            logger.info(f"\n{'='*50}\nStarting task: {task_name}\n{'='*50}")
            try:
                result = self.generate(
                    task_name=task_name,
                    model_name=model_name,
                    model_config=model_config,
                    output_dir=output_dir
                )
                all_results[task_name] = result
            except Exception as e:
                logger.error(f"Error generating for task {task_name}: {e}")
                all_results[task_name] = {'error': str(e)}
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Generate responses for CreativityPrism tasks")
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., aut, dat, ttcw) or "all" for all tasks')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--temperature', type=float, default=0.75,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='Maximum tokens to generate')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p sampling parameter')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to generate (None = all)')
    parser.add_argument('--output-dir', type=str, default='outputs/generations',
                       help='Output directory for generations')
    parser.add_argument('--config', type=str, default='config/tasks.yaml',
                       help='Path to task configuration file')
    parser.add_argument('--model-type', type=str, default='auto',
                       choices=['huggingface', 'openai', 'anthropic', 'vllm', 'auto'],
                       help='Model type/API to use')
    
    args = parser.parse_args()
    
    # Create model config
    model_config = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'top_p': args.top_p,
        'model_type': args.model_type
    }
    
    # Initialize pipeline
    pipeline = GenerationPipeline(config_path=args.config)
    
    # Generate
    if args.task.lower() == 'all':
        tasks = list(TASK_REGISTRY.keys())
        results = pipeline.generate_batch(
            tasks=tasks,
            model_name=args.model,
            model_config=model_config,
            output_dir=args.output_dir
        )
    else:
        results = pipeline.generate(
            task_name=args.task,
            model_name=args.model,
            model_config=model_config,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    
    # Print summary
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
