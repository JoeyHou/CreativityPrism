# Creative Bench - A Holistic Analysis on Creativity-as-a-Product in LLMs

```
creative_bench/
│── data/                        # Stores raw data and task-specific datasets
│   ├── raw/                     # Raw input data (maybe not included here)
│   ├── processed/               # Preprocessed data (JSON format)
│   ├── outputs/                 # Model-generated outputs
│   ├── evaluations/             # Evaluation results
│
│── configs/                     # Configurations for different tasks, models, and metrics
│   ├── default.json
│   ├── inference_configs.json
│   ├── evaluation_configs.json
│
│── src/                         # Core framework code
│   ├── prompt_engineering/      # Converts raw data into task-specific prompts
│   │   ├── __init__.py
│   │   ├── templates.py         # store all prompts
│   │   ├── functions.py         # helper functions for prompt engineering
│   │
│   ├── inference/               # Handles model inference using VLLM
│   │   ├── __init__.py
│   │   ├── inference_driver.py  # inference module (abstract class)
│   │   ├── task_inf_driver.py   # task specific inference driver 
│   │
│   ├── evaluation/              # Evaluates model outputs
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Main evaluation pipeline
│   │   ├── metrics.py           # Implements non-LLM-based metrics
│   │   ├── llm_based_eval.py    # Implements LLM-based evaluation
│   │
│   ├── utils/                   # Utility functions
│       ├── __init__.py
│       ├── helpers.py           # helper functions
│       ├── logging.py           # Custom logging functionality
│   
├── run_benchmark.py             # Runs a full benchmark pipeline
├── run_inference.py             # Runs inference only
├── run_evaluation.py            # Runs evaluation separately
├── run_testing.py               # Run some toy dataset for testing
│
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```


### Software Version
- `vllm`: [0.7.2](https://docs.vllm.ai/en/v0.7.2/getting_started/installation/index.html) (at least 0.7.0, for the support of deepseek-v3)
- `Python`: 3.9 – 3.12
- `cuda`: >= 12.1

### Code Structure
- scripts (e.g., `run_inference.py`): take in config, initialize inference driver (based on the task found in config)
- inference Driver (in `src/inference`)
    - initialize `vllm` 
    - create prompt (using template and functions in `src/prompt_engineering/`)
    - batched inference (`inference` func)
    - parse vllm output (`parse_vllm_outputs` func)
    - return final output in the same json file as input but with additional fields (`raw_output` and `parsed_output`)
- utils, prompt_engineering: helper functions, prompt templates

### Model Configuration
- `tensor_parallel_size`: `torch.cuda.device_count()` (all available GPU cards)
- `dtype`: `bfloat16` 

### Inference Settings
- `temperature`: 0.75
- `top_p`: 1
- `max_tokens`: 4096

### Running the Code
- Install dependencies: `pip install -r requirements.txt`
- All the scripts are in `\scripts`. Scripts with `_inference` are inference code for corresponding task; scripts with `_evaluation` are evaluation code. Please run inference before running evaluation