# MODELS=('Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct' 'OLMo-2-1124-7B-Instruct' 'Llama-3.1-8B-Instruct' 'OLMo-2-1124-13B-Instruct' 'OLMo-2-1124-13B-SFT' 'OLMo-2-1124-13B-DPO' 'Mistral-Small-24B-Instruct-2501' 'Qwen2.5-32B-Instruct' 'Mixtral-8x7B-Instruct-v0.1' 'Llama-3.3-70B-Instruct' 'Qwen2.5-72B-Instruct')
MODELS=('OLMo-2-1124-13B-DPO')

cd /playpen-ssd/abrei/creativity_benchmark/ttct

for MODEL in "${MODELS[@]}"
do
    echo "Showing evaluation results from $MODEL"
    python3 ./src/utils/show_results.py \
        -model_name "$MODEL" \
        -temp 1
done