# MODELS=('mistralai/Mistral-7B-Instruct-v0.3' 'Qwen/Qwen2.5-7B-Instruct' 'allenai/OLMo-2-1124-7B-Instruct' 'meta-llama/Llama-3.1-8B-Instruct' 'allenai/OLMo-2-1124-13B-Instruct' 'mistralai/Mistral-Small-24B-Instruct-2501' 'Qwen/Qwen2.5-32B-Instruct' 'mistralai/Mixtral-8x7B-Instruct-v0.1' 'meta-llama/Llama-3.3-70B-Instruct' 'Qwen/Qwen2.5-72B-Instruct'
#  'claude-3-7-sonnet-20250219' 'claude-3-5-haiku-20241022' 'gpt-4.1-2025-04-14' 'gpt-4.1-mini-2025-04-14' 'gemini-2.0-flash' 'deepseek-reasoner' 'deepseek-chat')

MODELS=('meta-llama/Llama-3.1-8B-Instruct' 'allenai/OLMo-2-1124-7B-Instruct')

# cd /playpen-ssd/abrei/creativity_benchmark/ttct
export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /playpen-ssd/abrei/creativity_benchmark/ttct
export PYTHONPATH=$PYTHONPATH:$(pwd)

for MODEL in "${MODELS[@]}"
do
    echo "Running $MODEL for inference"
    export HF_TOKEN=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
    python3 ./src/inference/ttct_inference.py \
        -model_name "$MODEL" \
        -temp 1
        # -prompt_formats "all"
done