# CreativityPrism Unified Evaluation Framework

A unified framework for generating and evaluating LLM responses across the CreativityPrism benchmark - a comprehensive evaluation of creativity across three dimensions (Quality, Novelty, Diversity) and three domains (Divergent Thinking, Creative Writing, Logical Reasoning).

## Overview

This framework provides:
1. **Unified Generation Script**: Execute any CreativityPrism task with any model
2. **Unified Evaluation Script**: Evaluate generated responses with task-specific metrics
3. **Modular Architecture**: Easy to extend with new tasks and metrics
4. **Multi-Model Support**: Works with OpenAI, Anthropic, Google, HuggingFace, and vLLM

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/creativity-prism-eval.git
cd creativity-prism-eval

# Install dependencies
pip install -r requirements.txt

# Set up API keys (if using proprietary models)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Project Structure

```
creativity_prism/
├── config/
│   └── tasks.yaml          # Task configurations
├── tasks/                  # Task implementations
│   ├── base.py
│   ├── aut.py              # Alternative Uses Task
│   ├── ttcw.py             # Creative Writing Task
│   └── ...                 # Other tasks
├── evaluators/             # Evaluation metrics
│   ├── base.py
│   ├── quality.py
│   ├── novelty.py
│   └── diversity.py
├── models/
│   └── model_interface.py  # Unified model interface
├── generate.py             # Generation script
├── evaluate.py             # Evaluation script
└── README.md
```

## Quick Start

### 1. Generate Responses

Generate responses for a single task:

```bash
python generate.py \
  --task aut \
  --model gpt-4 \
  --temperature 0.75 \
  --max-tokens 1024 \
  --output-dir outputs/generations
```

Generate for all tasks:

```bash
python generate.py \
  --task all \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-type huggingface \
  --output-dir outputs/generations
```

### 2. Evaluate Responses

Evaluate a single generation file:

```bash
python evaluate.py \
  --input outputs/generations/aut/gpt-4/generations_20241027_120000.jsonl \
  --evaluator-model gpt-4 \
  --output-dir outputs/evaluations
```

Evaluate all generations in a directory:

```bash
python evaluate.py \
  --input outputs/generations/ \
  --evaluator-model gpt-4 \
  --output-dir outputs/evaluations
```

### 3. Generate Leaderboard

Compute leaderboard from evaluation results:

```bash
python evaluate.py \
  --input outputs/evaluations/ \
  --leaderboard \
  --leaderboard-output outputs/leaderboard.json
```

## Supported Tasks

### Divergent Thinking Domain
- **AUT** (Alternative Uses Task): Generate creative alternative uses for common objects
- **DAT** (Divergent Association Task): Generate semantically distant words
- **TTCT** (Torrance Test): Multiple divergent thinking tasks

### Creative Writing Domain
- **TTCW** (Torrance Test of Creative Writing): Write creative stories from plots
- **Creative Short Story**: Generate creative stories with keyword constraints
- **Creativity Index**: Complete text prefixes creatively
- **CS4**: Write stories with multiple constraints

### Logical Reasoning Domain
- **NeoCoder**: Generate creative coding solutions
- **Creative Math**: Solve math problems with novel approaches

## Evaluation Dimensions

### Quality
Measures how well generated content fulfills task requirements:
- Story coherence
- Constraint satisfaction
- Code correctness
- Narrative completeness

### Novelty
Measures originality compared to existing content:
- Uniqueness of solutions
- Deviation from reference answers
- Semantic novelty
- Uncommon word usage

### Diversity
Measures variation among generated outputs:
- Lexical diversity
- Semantic distance
- Solution variety
- Emotional range

## Configuration

Edit `config/tasks.yaml` to customize:
- Task-specific parameters
- Generation configurations
- Metric settings
- Model defaults

Example configuration:

```yaml
tasks:
  aut:
    name: "Alternative Uses Task"
    domain: "divergent_thinking"
    num_uses: 10
    metrics:
      novelty: ["aut_score"]
    generation_config:
      temperature: 0.75
      max_tokens: 1024
```

## Adding New Tasks

1. Create a new task class in `tasks/`:

```python
from tasks.base import BaseTask, register_task

@register_task('my_task')
class MyTask(BaseTask):
    domain = 'creative_writing'
    
    def load_dataset(self):
        # Load your dataset
        return data
    
    def format_prompt(self, example):
        # Format prompt for model
        return prompt
    
    def get_metrics(self):
        return {
            'quality': ['my_metric'],
            'novelty': [],
            'diversity': []
        }
```

2. Add metrics in `evaluators/`:

```python
from evaluators.base import BaseEvaluator, register_evaluator

@register_evaluator('my_metric')
class MyMetricEvaluator(BaseEvaluator):
    dimension = 'quality'
    
    def evaluate(self, response, input_data, task=None):
        # Compute metric
        return score
```

3. Update `config/tasks.yaml` with task configuration

## Advanced Usage

### Using Different Model APIs

**OpenAI:**
```bash
python generate.py --task aut --model gpt-4 --model-type openai
```

**Anthropic:**
```bash
python generate.py --task aut --model claude-3-sonnet-20240229 --model-type anthropic
```

**HuggingFace:**
```bash
python generate.py --task aut --model meta-llama/Llama-3.1-8B-Instruct --model-type huggingface
```

**vLLM (for faster inference):**
```bash
python generate.py --task aut --model meta-llama/Llama-3.1-8B-Instruct --model-type vllm
```

### Batch Processing

Process multiple models across all tasks:

```bash
# Create a batch script
for model in "gpt-4" "claude-3-sonnet" "meta-llama/Llama-3.1-70B"; do
  python generate.py --task all --model "$model" --output-dir "outputs/generations"
  python evaluate.py --input "outputs/generations" --output-dir "outputs/evaluations"
done

# Generate final leaderboard
python evaluate.py --input outputs/evaluations --leaderboard
```

### Custom Evaluation

You can also use the pipeline programmatically:

```python
from generate import GenerationPipeline
from evaluate import EvaluationPipeline

# Generate
gen_pipeline = GenerationPipeline()
results = gen_pipeline.generate(
    task_name='aut',
    model_name='gpt-4',
    model_config={'temperature': 0.75, 'max_tokens': 1024}
)

# Evaluate
eval_pipeline = EvaluationPipeline()
eval_results = eval_pipeline.evaluate(
    generation_file=results['output_file'],
    evaluator_config={'judge_model': 'gpt-4'}
)

print(eval_results['aggregated_scores'])
```

## Aggregation Methodology

Scores are aggregated following the CreativityPrism methodology:

1. **Normalize** each metric to [0, 1] using min-max normalization
2. **Average** metrics within each dimension (quality, novelty, diversity)
3. **Compute overall** score as mean of dimension scores

## Citation

If you use this framework, please cite the CreativityPrism paper:

```bibtex
@article{creativityprism2024,
  title={CreativityPrism: A Holistic Benchmark for Large Language Model Creativity},
  author={...},
  journal={arXiv preprint arXiv:2510.20091},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Check the CreativityPrism project page: https://joeyhou.github.io/CreativityPrism/
- Read the paper: https://arxiv.org/abs/2510.20091
