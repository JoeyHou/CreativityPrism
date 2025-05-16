# MODELS=('Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct' 'OLMo-2-1124-7B-Instruct' 'Llama-3.1-8B-Instruct' 'OLMo-2-1124-13B-Instruct' 'OLMo-2-1124-13B-SFT' 'OLMo-2-1124-13B-DPO' 'Mistral-Small-24B-Instruct-2501' 'Qwen2.5-32B-Instruct' 'Mixtral-8x7B-Instruct-v0.1' 'Llama-3.3-70B-Instruct' 'Qwen2.5-72B-Instruct' 'claude-3-7-sonnet-20250219' 'claude-3-5-haiku-20241022' 'gpt-4.1-2025-04-14' 'gpt-4.1-mini-2025-04-14' 'gemini-2.0-flash' 'deepseek-reasoner' 'deepseek-chat')
MODELS=('OLMo-2-1124-13B-DPO')

export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /playpen-ssd/abrei/creativity_benchmark/ttct

for MODEL in "${MODELS[@]}"
do
    echo "Running $MODEL for evaluation"
    python3 ./src/evaluation/ttct_evaluation.py \
        -infer_model_name "$MODEL" \
        -eval_model_name "Qwen2.5-72B-Instruct" \
        -temp 1 \
        -cache_dir "/playpen-ssd/pretrained_models/"
done