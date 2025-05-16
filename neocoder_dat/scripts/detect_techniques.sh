declare -a arr=("results/neocoder/inference/deepseek-chat_sample=199_dp=5_0.json" 
                "results/neocoder/inference/deepseek-chat_sample=199_dp=5_25.json"
                "results/neocoder/inference/deepseek-chat_sample=199_dp=5_50.json"
                "results/neocoder/inference/deepseek-chat_sample=199_dp=5_100.json"
                "results/neocoder/inference/OLMo-2-1124-13B-Instruct_sample=199_dp=5_0.json"
                "results/neocoder/inference/OLMo-2-1124-13B-Instruct_sample=199_dp=5_25.json"
                "results/neocoder/inference/OLMo-2-1124-13B-Instruct_sample=199_dp=5_50.json"
                "results/neocoder/inference/OLMo-2-1124-13B-Instruct_sample=199_dp=5_100.json"
                "results/neocoder/inference/OLMo-2-1124-13B-SFT_sample=199_dp=5_75.json"
                "results/neocoder/inference/OLMo-2-1124-13B-DPO_sample=199_dp=5_75.json"
                )

for result in "${arr[@]}"
do
    echo "Processing $result"
    python steps/evaluate_neogauge.py \
        --task detection \
        --inference-result-path $result \
        --human-solution-path datasets/CodeForce/NeoCoder/human_solutions.json
done
