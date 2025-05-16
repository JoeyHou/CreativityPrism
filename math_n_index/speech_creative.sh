#!/bin/bash
#SBATCH --job-name=deepseek-reasoner_speech
#SBATCH --output=index_logs/deepseek-reasoner-speech_%j.out
#SBATCH --error=index_logs/deepseek-reasoner-speech_%j.err
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
source # path
conda activate creative

for MIN_NGRAM in {5..12}; do # range of L
    echo "Running with min_ngram=$MIN_NGRAM at $(date)"
    
    python -m src.evaluation.evaluation_creative_index_parr \
        --task deepseek-reasoner-speech-dolma \
        --data data/outputs/creative_index/deepseek-reasoner/speech.json \
        --output_dir data/evaluation_dolma/creative_index_evaluations/creative_index_exact/deepseek-reasoner/speech \
        --min_ngram $MIN_NGRAM \
        --subset 100 \
        --lm_tokenizer \
        --num_workers 8

    echo "Finished min_ngram=$MIN_NGRAM at $(date)"
done

# Log the end time
echo "Job finished at: $(date)"
