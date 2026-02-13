export VLLM_WORKER_MULTIPROC_METHOD=spawn

# running
python run_evaluation.py configs/aut/evaluation/olmo_13b_sft.json
python run_evaluation.py configs/aut/evaluation/olmo_13b_dpo.json
python run_evaluation.py configs/ttcw/evaluation/olmo_13b_sft.json
python run_evaluation.py configs/ttcw/evaluation/olmo_13b_dpo.json
# python run_evaluation.py configs/creative_short/olmo_sft.json