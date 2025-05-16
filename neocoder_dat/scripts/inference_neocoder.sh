python steps/inference_dp.py \
        --dataset-path datasets/CodeForce/NeoCoder/NeoCoder.json \
        --model-name allenai/OLMo-2-1124-13B-SFT \
        --dp-rounds 5 \
        --batch-size 1 \
        --output-dir results/neocoder/inference \
        --overwrite \
        --temperature 0.75 \
        --top-p 1 \
        --max-tokens 1024
