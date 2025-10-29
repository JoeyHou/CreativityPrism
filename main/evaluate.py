#!/usr/bin/env python3
"""
Unified evaluation script for CreativityPrism benchmark.
Evaluates generated responses across quality, novelty, and diversity dimensions.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging
import numpy as np

from tasks import TASK_REGISTRY
from evaluators import EVALUATOR_REGISTRY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Unified pipeline for evaluating generated responses across all metrics."""
    
    def __init__(self, config_path: str = "config/tasks.yaml"):
        """Initialize the evaluation pipeline."""
        self.config = self._load_config(config_path)
        self.dimension_map = self._build_dimension_map()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load evaluation configurations from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_dimension_map(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build mapping of tasks to their metrics by dimension.
        
        Returns:
            {
                'task_name': {
                    'quality': ['metric1', 'metric2'],
                    'novelty': ['metric3'],
                    'diversity': ['metric4']
                }
            }
        """
        dimension_map = {}
        for task_name, task_config in self.config.get('tasks', {}).items():
            metrics = task_config.get('metrics', {})
            dimension_map[task_name] = {
                'quality': metrics.get('quality', []),
                'novelty': metrics.get('novelty', []),
                'diversity': metrics.get('diversity', [])
            }
        return dimension_map
    
    def load_generations(self, generation_file: str) -> List[Dict]:
        """Load generated responses from JSONL file."""
        generations = []
        with open(generation_file, 'r') as f:
            for line in f:
                generations.append(json.loads(line))
        return generations
    
    def evaluate(
        self,
        generation_file: str,
        task_name: str = None,
        output_dir: str = "outputs/evaluations",
        evaluator_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated responses using task-specific metrics.
        
        Args:
            generation_file: Path to JSONL file with generated responses
            task_name: Task name (auto-detected if None)
            output_dir: Directory to save evaluation results
            evaluator_config: Configuration for evaluators (e.g., LLM judge model)
            
        Returns:
            Dictionary containing evaluation results and aggregated scores
        """
        # Load generations
        generations = self.load_generations(generation_file)
        logger.info(f"Loaded {len(generations)} generations from {generation_file}")
        
        # Auto-detect task if not provided
        if task_name is None:
            task_name = generations[0]['task']
        logger.info(f"Evaluating task: {task_name}")
        
        # Initialize task and evaluators
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Task '{task_name}' not found")
        
        task_class = TASK_REGISTRY[task_name]
        task = task_class()
        
        # Get metric configuration for this task
        metrics_config = self.dimension_map.get(task_name, {})
        
        # Initialize evaluators
        evaluator_config = evaluator_config or {}
        evaluators = {}
        for dimension in ['quality', 'novelty', 'diversity']:
            metric_names = metrics_config.get(dimension, [])
            if metric_names:
                for metric_name in metric_names:
                    if metric_name in EVALUATOR_REGISTRY:
                        evaluators[metric_name] = EVALUATOR_REGISTRY[metric_name](**evaluator_config)
        
        logger.info(f"Initialized {len(evaluators)} evaluators: {list(evaluators.keys())}")
        
        # Evaluate each generation
        results = []
        for idx, gen in enumerate(generations):
            logger.info(f"Evaluating {idx+1}/{len(generations)}")
            
            # Compute metrics
            scores = {}
            for metric_name, evaluator in evaluators.items():
                try:
                    score = evaluator.evaluate(
                        response=gen['response'],
                        input_data=gen['input'],
                        task=task
                    )
                    scores[metric_name] = score
                except Exception as e:
                    logger.error(f"Error computing {metric_name}: {e}")
                    scores[metric_name] = None
            
            result = {
                'example_id': gen.get('example_id', idx),
                'scores': scores
            }
            results.append(result)
        
        # Aggregate scores by dimension
        aggregated = self._aggregate_scores(results, metrics_config)
        
        # Save evaluation results
        output_path = Path(output_dir) / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        gen_file_name = Path(generation_file).stem
        output_file = output_path / f"eval_{gen_file_name}.json"
        
        eval_output = {
            'task': task_name,
            'generation_file': generation_file,
            'num_examples': len(results),
            'raw_results': results,
            'aggregated_scores': aggregated,
            'metadata': {
                'evaluator_config': evaluator_config,
                'metrics_config': metrics_config
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(eval_output, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")
        
        return eval_output
    
    def _aggregate_scores(
        self,
        results: List[Dict],
        metrics_config: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Aggregate raw scores into dimension-level and overall scores.
        
        Follows CreativityPrism aggregation methodology:
        1. Normalize each metric to [0, 1]
        2. Average metrics within each dimension
        3. Compute overall score as mean of dimension scores
        """
        # Collect all scores by metric
        metric_scores = defaultdict(list)
        for result in results:
            for metric_name, score in result['scores'].items():
                if score is not None:
                    metric_scores[metric_name].append(score)
        
        # Compute normalized scores for each metric
        normalized_metrics = {}
        for metric_name, scores in metric_scores.items():
            if scores:
                # Min-max normalization
                min_score = self._get_metric_min(metric_name)
                max_score = self._get_metric_max(metric_name)
                
                normalized = []
                for score in scores:
                    if max_score - min_score > 0:
                        norm_score = (score - min_score) / (max_score - min_score)
                    else:
                        norm_score = score
                    normalized.append(np.clip(norm_score, 0, 1))
                
                normalized_metrics[metric_name] = np.mean(normalized)
        
        # Aggregate by dimension
        dimension_scores = {}
        for dimension in ['quality', 'novelty', 'diversity']:
            metric_names = metrics_config.get(dimension, [])
            if metric_names:
                dim_scores = [normalized_metrics.get(m) for m in metric_names if m in normalized_metrics]
                if dim_scores:
                    dimension_scores[dimension] = float(np.mean(dim_scores))
        
        # Overall score
        if dimension_scores:
            overall_score = float(np.mean(list(dimension_scores.values())))
        else:
            overall_score = 0.0
        
        return {
            'overall': overall_score,
            **dimension_scores,
            'metrics': normalized_metrics
        }
    
    def _get_metric_min(self, metric_name: str) -> float:
        """Get minimum possible value for a metric."""
        # Define metric ranges (customize based on actual metrics)
        metric_ranges = {
            'aut_score': 1.0,
            'dat_score': 0.0,
            'fluency': 1.0,
            'originality': 1.0,
            'flexibility': 1.0,
            'elaboration': 1.0,
            'coherence': 0.0,
            'constraint_satisfaction': 0.0,
            'l_uniqueness': 0.0,
            # Add more as needed
        }
        return metric_ranges.get(metric_name, 0.0)
    
    def _get_metric_max(self, metric_name: str) -> float:
        """Get maximum possible value for a metric."""
        # Define metric ranges
        metric_ranges = {
            'aut_score': 5.0,
            'dat_score': 1.0,
            'fluency': 5.0,
            'originality': 5.0,
            'flexibility': 5.0,
            'elaboration': 5.0,
            'coherence': 1.0,
            'constraint_satisfaction': 1.0,
            'l_uniqueness': 1.0,
            # Add more as needed
        }
        return metric_ranges.get(metric_name, 1.0)
    
    def evaluate_batch(
        self,
        generation_files: List[str],
        output_dir: str = "outputs/evaluations",
        evaluator_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple generation files."""
        all_results = {}
        
        for gen_file in generation_files:
            logger.info(f"\n{'='*50}\nEvaluating: {gen_file}\n{'='*50}")
            try:
                result = self.evaluate(
                    generation_file=gen_file,
                    output_dir=output_dir,
                    evaluator_config=evaluator_config
                )
                all_results[gen_file] = result
            except Exception as e:
                logger.error(f"Error evaluating {gen_file}: {e}")
                all_results[gen_file] = {'error': str(e)}
        
        return all_results
    
    def compute_leaderboard(
        self,
        eval_files: List[str],
        output_file: str = "outputs/leaderboard.json"
    ) -> Dict[str, Any]:
        """
        Compute leaderboard from multiple evaluation files.
        
        Args:
            eval_files: List of evaluation JSON files
            output_file: Path to save leaderboard
            
        Returns:
            Leaderboard with model rankings across all dimensions
        """
        leaderboard = defaultdict(lambda: defaultdict(list))
        
        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            model_name = eval_data.get('model', 'unknown')
            task_name = eval_data['task']
            scores = eval_data['aggregated_scores']
            
            # Collect scores by dimension
            for dimension in ['overall', 'quality', 'novelty', 'diversity']:
                if dimension in scores:
                    leaderboard[model_name][dimension].append({
                        'task': task_name,
                        'score': scores[dimension]
                    })
        
        # Compute average scores per model
        rankings = {}
        for model_name, dimensions in leaderboard.items():
            rankings[model_name] = {}
            for dimension, task_scores in dimensions.items():
                if task_scores:
                    avg_score = np.mean([ts['score'] for ts in task_scores])
                    rankings[model_name][dimension] = float(avg_score)
        
        # Sort by overall score
        sorted_rankings = sorted(
            rankings.items(),
            key=lambda x: x[1].get('overall', 0),
            reverse=True
        )
        
        leaderboard_output = {
            'rankings': dict(sorted_rankings),
            'num_models': len(rankings)
        }
        
        # Save leaderboard
        with open(output_file, 'w') as f:
            json.dump(leaderboard_output, f, indent=2)
        
        logger.info(f"Saved leaderboard to {output_file}")
        
        return leaderboard_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate CreativityPrism generations")
    parser.add_argument('--input', type=str, required=True,
                       help='Path to generation JSONL file or directory')
    parser.add_argument('--task', type=str, default=None,
                       help='Task name (auto-detected if None)')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluations',
                       help='Output directory for evaluation results')
    parser.add_argument('--config', type=str, default='config/tasks.yaml',
                       help='Path to task configuration file')
    parser.add_argument('--evaluator-model', type=str, default='gpt-4',
                       help='Model to use for LLM-as-judge metrics')
    parser.add_argument('--leaderboard', action='store_true',
                       help='Compute leaderboard from evaluation files')
    parser.add_argument('--leaderboard-output', type=str, default='outputs/leaderboard.json',
                       help='Output file for leaderboard')
    
    args = parser.parse_args()
    
    # Evaluator config
    evaluator_config = {
        'judge_model': args.evaluator_model
    }
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config_path=args.config)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if args.leaderboard:
        # Compute leaderboard
        if input_path.is_dir():
            eval_files = list(input_path.glob('**/eval_*.json'))
        else:
            eval_files = [str(input_path)]
        
        leaderboard = pipeline.compute_leaderboard(
            eval_files=eval_files,
            output_file=args.leaderboard_output
        )
        
        # Print leaderboard
        print("\n" + "="*50)
        print("LEADERBOARD")
        print("="*50)
        for rank, (model, scores) in enumerate(leaderboard['rankings'].items(), 1):
            print(f"{rank}. {model}")
            for dim, score in scores.items():
                print(f"   {dim}: {score:.4f}")
    
    elif input_path.is_file():
        # Evaluate single file
        result = pipeline.evaluate(
            generation_file=str(input_path),
            task_name=args.task,
            output_dir=args.output_dir,
            evaluator_config=evaluator_config
        )
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Task: {result['task']}")
        print(f"Examples: {result['num_examples']}")
        print("\nAggregated Scores:")
        for key, value in result['aggregated_scores'].items():
            if key != 'metrics':
                print(f"  {key}: {value:.4f}")
    
    else:
        # Evaluate all files in directory
        gen_files = list(input_path.glob('**/*.jsonl'))
        results = pipeline.evaluate_batch(
            generation_files=[str(f) for f in gen_files],
            output_dir=args.output_dir,
            evaluator_config=evaluator_config
        )
        
        print(f"\nEvaluated {len(results)} generation files")


if __name__ == '__main__':
    main()
