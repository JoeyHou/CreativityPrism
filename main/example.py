#!/usr/bin/env python3
"""
Example script demonstrating the complete CreativityPrism evaluation pipeline.
"""

from generate import GenerationPipeline
from evaluate import EvaluationPipeline
import json
from pathlib import Path


def run_single_task_evaluation():
    """Example: Generate and evaluate a single task with one model."""
    print("=" * 60)
    print("EXAMPLE 1: Single Task Evaluation")
    print("=" * 60)
    
    # Initialize pipelines
    gen_pipeline = GenerationPipeline()
    eval_pipeline = EvaluationPipeline()
    
    # Configure model
    model_config = {
        'temperature': 0.75,
        'max_tokens': 1024,
        'top_p': 1.0,
        'model_type': 'openai'
    }
    
    # Generate responses
    print("\n1. Generating responses for AUT task...")
    gen_results = gen_pipeline.generate(
        task_name='aut',
        model_name='gpt-4',
        model_config=model_config,
        num_samples=5  # Generate for only 5 examples
    )
    
    print(f"   Generated {gen_results['num_generated']} responses")
    print(f"   Saved to: {gen_results['output_file']}")
    
    # Evaluate responses
    print("\n2. Evaluating generated responses...")
    eval_results = eval_pipeline.evaluate(
        generation_file=gen_results['output_file'],
        evaluator_config={'judge_model': 'gpt-4'}
    )
    
    # Print results
    print("\n3. Results:")
    print(f"   Overall Score: {eval_results['aggregated_scores']['overall']:.4f}")
    for dim in ['quality', 'novelty', 'diversity']:
        if dim in eval_results['aggregated_scores']:
            print(f"   {dim.capitalize()}: {eval_results['aggregated_scores'][dim]:.4f}")


def run_multi_task_comparison():
    """Example: Compare multiple models across multiple tasks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Model Comparison")
    print("=" * 60)
    
    gen_pipeline = GenerationPipeline()
    eval_pipeline = EvaluationPipeline()
    
    # Define models to compare
    models = [
        {'name': 'gpt-4', 'type': 'openai'},
        {'name': 'claude-3-sonnet-20240229', 'type': 'anthropic'},
    ]
    
    # Define tasks to test
    tasks = ['aut', 'ttcw']
    
    eval_files = []
    
    # Generate and evaluate for each model and task
    for model in models:
        print(f"\nProcessing model: {model['name']}")
        
        model_config = {
            'temperature': 0.75,
            'max_tokens': 2048,
            'model_type': model['type']
        }
        
        for task in tasks:
            print(f"  Task: {task}")
            
            # Generate
            gen_results = gen_pipeline.generate(
                task_name=task,
                model_name=model['name'],
                model_config=model_config,
                num_samples=3
            )
            
            # Evaluate
            eval_results = eval_pipeline.evaluate(
                generation_file=gen_results['output_file'],
                evaluator_config={'judge_model': 'gpt-4'}
            )
            
            eval_files.append(gen_results['output_file'].replace('generations', 'evaluations').replace('.jsonl', '.json'))
            
            print(f"    Overall: {eval_results['aggregated_scores']['overall']:.4f}")
    
    # Generate leaderboard
    print("\n4. Generating leaderboard...")
    leaderboard = eval_pipeline.compute_leaderboard(
        eval_files=eval_files,
        output_file='outputs/leaderboard_example.json'
    )
    
    print("\n5. Leaderboard:")
    for rank, (model_name, scores) in enumerate(leaderboard['rankings'].items(), 1):
        print(f"   {rank}. {model_name}: {scores.get('overall', 0):.4f}")


def run_batch_processing():
    """Example: Batch process all tasks for a model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing All Tasks")
    print("=" * 60)
    
    gen_pipeline = GenerationPipeline()
    eval_pipeline = EvaluationPipeline()
    
    model_config = {
        'temperature': 0.75,
        'max_tokens': 2048,
        'model_type': 'openai'
    }
    
    # Generate for all tasks
    print("\n1. Generating responses for all tasks...")
    all_tasks = ['aut', 'ttcw']  # Add more tasks as needed
    
    gen_results = gen_pipeline.generate_batch(
        tasks=all_tasks,
        model_name='gpt-4',
        model_config=model_config
    )
    
    # Collect generation files
    gen_files = [result['output_file'] for task, result in gen_results.items() if 'output_file' in result]
    
    # Evaluate all
    print("\n2. Evaluating all generations...")
    eval_results = eval_pipeline.evaluate_batch(
        generation_files=gen_files,
        evaluator_config={'judge_model': 'gpt-4'}
    )
    
    # Print summary
    print("\n3. Summary by Task:")
    for task, result in eval_results.items():
        if 'error' not in result:
            overall = result['aggregated_scores']['overall']
            print(f"   {Path(task).parent.name}: {overall:.4f}")


def analyze_results():
    """Example: Analyze and visualize evaluation results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Analyzing Results")
    print("=" * 60)
    
    # Load evaluation results
    eval_file = "outputs/evaluations/aut/eval_generations_*.json"
    
    try:
        # Find the most recent evaluation file
        from glob import glob
        eval_files = glob(eval_file)
        if not eval_files:
            print("No evaluation files found. Run previous examples first.")
            return
        
        latest_eval = sorted(eval_files)[-1]
        
        with open(latest_eval, 'r') as f:
            results = json.load(f)
        
        print(f"\nAnalyzing: {latest_eval}")
        print(f"Task: {results['task']}")
        print(f"Number of examples: {results['num_examples']}")
        
        # Show dimension breakdown
        print("\nDimension Scores:")
        for dim in ['quality', 'novelty', 'diversity']:
            if dim in results['aggregated_scores']:
                score = results['aggregated_scores'][dim]
                print(f"  {dim.capitalize()}: {score:.4f} {'★' * int(score * 5)}")
        
        # Show individual metrics
        print("\nIndividual Metrics:")
        for metric, score in results['aggregated_scores'].get('metrics', {}).items():
            print(f"  {metric}: {score:.4f}")
        
        # Show example responses
        print("\nSample Response:")
        if results['raw_results']:
            example = results['raw_results'][0]
            print(f"  Example ID: {example['example_id']}")
            print(f"  Scores: {example['scores']}")
    
    except Exception as e:
        print(f"Error analyzing results: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CreativityPrism Evaluation Framework - Examples")
    print("=" * 60)
    
    # Note: Uncomment the examples you want to run
    # Make sure you have API keys set up for the models you use
    
    try:
        # Example 1: Single task, single model
        # run_single_task_evaluation()
        
        # Example 2: Multiple models comparison
        # run_multi_task_comparison()
        
        # Example 3: Batch processing
        # run_batch_processing()
        
        # Example 4: Analyze results
        # analyze_results()
        
        print("\n" + "=" * 60)
        print("Examples completed! Check the outputs/ directory for results.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set up API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        print("2. Installed all dependencies (pip install -r requirements.txt)")
        print("3. Prepared the necessary data files in the data/ directory")


if __name__ == '__main__':
    # Uncomment examples in main() to run them
    print("\nThis is an example script. Edit the main() function to run specific examples.")
    print("Make sure to set up your API keys and data files first.\n")
    
    # Show what each example does
    print("Available examples:")
    print("1. run_single_task_evaluation() - Generate and evaluate one task")
    print("2. run_multi_task_comparison() - Compare multiple models")
    print("3. run_batch_processing() - Process all tasks at once")
    print("4. analyze_results() - Analyze evaluation results")
    print("\nUncomment the examples you want to run in the main() function.")
