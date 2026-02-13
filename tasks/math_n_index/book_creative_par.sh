#!/bin/bash
#SBATCH --job-name=OLMo2-13B-sft_book
#SBATCH --output=index_logs/OLMo2-13B-sft-book_%j.out
#SBATCH --error=index_logs/OLMo2-13B-sft-book_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --nodes=1

# Ensure log directory exists
mkdir -p log

# Log the start time
echo "Job started at: $(date)"

# Activate your conda environment
source # activate your conda env 
conda activate creative

# Loop over min_ngram values from 5 to 12
for MIN_NGRAM in {5..12}; do # range of L
    echo "Running with min_ngram=$MIN_NGRAM at $(date)"
    
    python -m src.evaluation.evaluation_creative_index_parr \
        --task OLMo2-13B-sft-book-dolma \
        --data data/outputs/new_index/OLMo2-13B-sft/book.json \
        --output_dir data/OLMo2-13B-sft \
        --min_ngram $MIN_NGRAM \
        --subset 100 \
        --lm_tokenizer \
        --num_workers 8

    echo "Finished min_ngram=$MIN_NGRAM at $(date)"
done

echo "Job finished at: $(date)"
