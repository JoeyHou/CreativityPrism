# Creative Bench - TTCT

```
creative_bench/
│── data/                        # Stores raw data and task-specific datasets
│   ├── raw/                     # Raw input data (maybe not included here)
│   ├── processed/               # Preprocessed data (JSON format)
│   ├── outputs/                 # Model-generated outputs
│   ├── evaluations/             # Evaluation results
│
│── src/                         # Core framework code
│   ├── inference/               # Handles model inference using VLLM
│   │   ├── __init__.py
│   │   ├── ttct_inference.py    # Main inference module
│   │
│   ├── evaluation/              # Evaluates model outputs
│   │   ├── __init__.py
│   │   ├── ttct_evaluation.py   # Main evaluation pipeline
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── helpers.py           # Handles saving/loading of files
│   │   ├── download_model.py    # Handles downloading open-source model weights
│   │   ├── api_wrapper.py       # Set up APIs
│   │   ├── run_api.py           # Handles calls to APIs
│   │   ├── show_results.py      # Displays evaluation results
│   │
│── scripts/                     # Entry points for running tasks
│   ├── download_models.sh       # Download model weights to local directory
│   ├── run_inference.sh         # Runs inference only
│   ├── run_evaluation.py        # Runs evaluation only
│   ├── show_results.sh          # Display evaluation results 
│
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```

## Set Up

All code is run from shell scripts in directory ```script/```. Before running from the commandline, please check all variables in the shell script to ensure they are set according to your needs.

**To download models locally:**

```
cd ttct/scripts/
chmod +x download_models.sh
./download_models.sh
```

**To perform inference:**

```
chmod +x run_inference.sh
./run_inference.sh
```

**To perform evaluations:**

```
chmod +x run_evaluation.sh
./run_evaluation.sh
```

**To show results:**

```
chmod +x show_results.sh
./show_results.sh
```