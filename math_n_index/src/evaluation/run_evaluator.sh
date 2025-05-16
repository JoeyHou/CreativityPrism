#!/bin/bash
#SBATCH --job-name=my_vllm_job          # Job name
#SBATCH --output=logs/job_%j.out        # Standard output log file
#SBATCH --error=logs/job_%j.err         # Error log file

# Load environment
source activate creative  # Activate your Conda environment

# Define evaluators
EVALUATORS=("Llama-31-8B-Instruct" "Qwen2.5-7B-Instruct")

for EVAL_MODEL in "${EVALUATORS[@]}"; do
    echo "Starting evaluation for model: $EVAL_MODEL"
    python -m src.evaluation.creative_math_evaluation --eval_lm "$EVAL_MODEL"

    # Cleanup memory and processes after each evaluation
    echo "Clearing memory and killing workers..."
    pkill -u $USER -f "vllm_worker"
    pkill -u $USER -f "python"
    
    echo "Evaluation completed for model: $EVAL_MODEL"
done

