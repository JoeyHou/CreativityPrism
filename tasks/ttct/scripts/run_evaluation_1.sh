# MODELS=('Mistral-7B-Instruct-v0.3' 'Qwen2.5-7B-Instruct' 'OLMo-2-1124-7B-Instruct' 'Llama-3.1-8B-Instruct' 'OLMo-2-1124-13B-Instruct' 'OLMo-2-1124-13B-SFT' 'OLMo-2-1124-13B-DPO' 'Mistral-Small-24B-Instruct-2501' 'Qwen2.5-32B-Instruct' 'Mixtral-8x7B-Instruct-v0.1' 'Llama-3.3-70B-Instruct' 'Qwen2.5-72B-Instruct' 'claude-3-7-sonnet-20250219' 'claude-3-5-haiku-20241022' 'gpt-4.1-2025-04-14' 'gpt-4.1-mini-2025-04-14' 'gemini-2.0-flash' 'deepseek-reasoner' 'deepseek-chat')
# MODELS=('OLMo-2-1124-13B-DPO')

# Joey edits
# MODELS=('Qwen2.5-7B-Instruct' 'gpt-4.1-2025-04-14' 'OLMo-2-1124-13B-Instruct')
# MODELS=('claude-3-7-sonnet-20250219' 'claude-3-haiku-20240307' 'gpt-4.1-mini-2025-04-14')
MODELS=('Mixtral-8x7B-Instruct-v0.1' 'OLMo-2-1124-7B-Instruct' 'OLMo-2-1124-13B-Instruct' 'Qwen2.5-7B-Instruct' 'Qwen2.5-32B-Instruct' 'Qwen2.5-72B-Instruct' 'Llama-3.3-70B-Instruct' 'Mistral-7B-Instruct-v0.3')
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /playpen-ssd/abrei/creativity_benchmark/ttct
export PYTHONPATH=$PYTHONPATH:$(pwd)

################ Qwen 72B Judge #################
EVAL_MODEL='Qwen/Qwen2.5-72B-Instruct'
for MODEL in "${MODELS[@]}"
do
    echo "Running $MODEL for evaluation"

    # Joey: set temp to -1 for llm-judge v.s human eval
    python3 ./src/evaluation/ttct_evaluation.py \
        -infer_model_name "$MODEL" \
        -eval_model_name "$EVAL_MODEL" \
        -run_id "temp_1_qwen"\
        -cache_dir /ihome/xli/joh227/ix_dir/huggingface/hub\
        -summary true
done

MODELS=('Mistral-Small-24B-Instruct-2501' 'Mixtral-8x7B-Instruct-v0.1' 'OLMo-2-1124-7B-Instruct' 'OLMo-2-1124-13B-Instruct' 'Qwen2.5-7B-Instruct' 'Qwen2.5-32B-Instruct' 'Qwen2.5-72B-Instruct' 'Llama-3.3-70B-Instruct' 'Mistral-7B-Instruct-v0.3')

################# GPT4.1 Judge #################
EVAL_MODEL='gpt-4.1-2025-04-14'
for MODEL in "${MODELS[@]}"
do
    echo "Running $MODEL for evaluation"

    # Joey: set temp to -1 for llm-judge v.s human eval
    python3 ./src/evaluation/ttct_evaluation.py \
        -infer_model_name "$MODEL" \
        -eval_model_name "$EVAL_MODEL" \
        -run_id "temp_1_gpt41"\
        -cache_dir /ihome/xli/joh227/ix_dir/huggingface/hub\
        -summary true
        # -pairwise true
        # -cache_dir "/playpen-ssd/pretrained_models/"
        # -temp -1 \
done

# ################ Gemini Judge #################
# EVAL_MODEL='gemini-2.5-flash'
# for MODEL in "${MODELS[@]}"
# do
#     echo "Running $MODEL for evaluation"

#     # Joey: set temp to -1 for llm-judge v.s human eval
#     python3 ./src/evaluation/ttct_evaluation.py \
#         -infer_model_name "$MODEL" \
#         -eval_model_name "$EVAL_MODEL" \
#         -run_id "gemini_judge_summary"\
#         -cache_dir /ihome/xli/joh227/ix_dir/huggingface/hub\
#         -summary true
# done