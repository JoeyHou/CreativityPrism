#!/bin/bash
#SBATCH --job-name=qwen-7b-instruct_poem
#SBATCH --output=index_logs/qwen-7b-instruct_%j.out
#SBATCH --error=index_logs/qwen-7b-instruct-poem_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --nodes=1

# Ensure log directory exists
mkdir -p log

# Log the start time
echo "Job started at: $(date)"

# activate your conda environment
source # path
conda activate creative

for MIN_NGRAM in {5..12}; do # range of L
    echo "Running with min_ngram=$MIN_NGRAM at $(date)"

    python -m src.evaluation.evaluation_creative_index_parr \
        --task qwen-7b-instruct-poem-dolma \
        --data data/outputs/new_index/qwen-7b-instruct/poem.json \
        --output_dir data/qwen-7b-instruct/poem \
        --min_ngram $MIN_NGRAM \
        --subset 100 \
        --lm_tokenizer \
        --num_workers 8

    echo "Finished min_ngram=$MIN_NGRAM at $(date)"
done

# Log the end time
echo "Job finished at: $(date)"
