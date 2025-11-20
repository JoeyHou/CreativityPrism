export HF_HOME=/scratch/dkhasha1/bzhang90/huggingface
export HF_DATASETS_CACHE=/scratch/dkhasha1/bzhang90/huggingface/datasets
export HF_HUB_CACHE=/scratch/dkhasha1/bzhang90/huggingface/hub

CUDA_VISIBLE_DEVICES=0,1 python generate_sh.py --task aut --model qwen_32b_instruct --config /scratch/dkhasha1/bzhang90/creativityprism/aut_ttcw_cshort/configs/aut/inference/qwen_1.5b.json \
