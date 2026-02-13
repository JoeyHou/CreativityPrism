export HF_HOME=/scratch/dkhasha1/bzhang90/huggingface
export HF_DATASETS_CACHE=/scratch/dkhasha1/bzhang90/huggingface/datasets
export HF_HUB_CACHE=/scratch/dkhasha1/bzhang90/huggingface/hub

# run eval for aut_ttcw_cshort
# CUDA_VISIBLE_DEVICES=0 python generate_sh.py --task aut --model qwen_32b_instruct --config /scratch/dkhasha1/bzhang90/creativityprism/aut_ttcw_cshort/configs/aut/inference/qwen_1.5b.json \

# cs4
CUDA_VISIBLE_DEVICES=0 python generate_sh.py --task cs4 --config /scratch/dkhasha1/bzhang90/creativityprism/cs4/story_generator/configs/qwen_7b.json

# ttct: no configs needed as we only need to change model names here
# CUDA_VISIBLE_DEVICES=0 python generate_sh.py --task ttct 

