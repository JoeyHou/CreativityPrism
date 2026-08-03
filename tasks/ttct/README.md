# Creative Bench - TTCT

## Question types: 7 shipped, 5 scored

The dataset ships **700 items across 7 question types** (100 each): `1_unusual_uses`,
`2_consequences`, `3_just_suppose`, `4_situation`, `5_common_problems`, `6_improvement`,
`7_story`.

**Only 5 of them are scored by default**: `1_unusual_uses`, `2_consequences`, `4_situation`,
`5_common_problems`, `6_improvement`. The LLM-judge rubric was aligned against human ratings for
those 5 only, so `3_just_suppose` and `7_story` are shipped for completeness but are marked
`skip` and left unscored. The list lives in `DEFAULT_SUBSET`, defined identically in
`src/inference/ttct_inference.py` and `src/evaluation/ttct_evaluation.py`; pass `-subset` to
override it.

## `--limit N`

Evaluation asserts the inference output has exactly one row per `data/processed/basefile.csv`
row (700), so `--limit N` does **not** truncate the file. It keeps all 700 rows and queries only
the **first N items of each scored question type** — `5 x N` model calls, with the rest marked
`skip` so the judge does not score them either. `--limit 2` therefore scores 10 items, not 2.

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