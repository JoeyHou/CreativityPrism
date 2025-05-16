python steps/inference_dat.py \
        --model-name allenai/OLMo-2-1124-13B-SFT \
        --repeat 100 \
        --output-dir results/dat/inference \
        --overwrite \
        --temperature 0.75 \
        --top-p 1 \
        --max-tokens 4096
