# python evaluation/story_quality_eval.py --run_id=cs4_v1/llama3_8b_instr 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/llama3_70b_instruct 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/mistral_7b_instr 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/mistral_small_24b 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/mixtral_8x7b 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/olmo_7b 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/olmo_13b 

# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_7b_instruct 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_32b_instruct 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/gpt_4.1
# python evaluation/story_quality_eval.py --run_id=cs4_v1/gpt_4.1_mini
# python evaluation/story_quality_eval.py --run_id=cs4_v1/deepseek_r1
# python evaluation/story_quality_eval.py --run_id=cs4_v1/deepseek_v3
# python evaluation/story_quality_eval.py --run_id=cs4_v1/gemini_2_flash

# python evaluation/story_quality_eval.py --run_id=cs4_v1/claude_3_haiku 
# python evaluation/story_quality_eval.py --run_id=cs4_v1/claude_37_sonnet 

# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_72b_instruct

# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_72b_instruct_qwen




# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_72b_instruct_llama
# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_72b_instruct_qwen
# python evaluation/constraint_satisfaction.py --run_id=cs4_v1/qwen_72b_instruct_llama --use_vllm


# python evaluation/constraint_satisfaction.py --run_id=cs4_v1/deepseek_r1
# python evaluation/story_quality_eval.py --run_id=cs4_v1/qwen_72b_instruct



######### temperature varying experiments #########
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/olmo_0
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/olmo_1
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/olmo_05
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/olmo_025
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/qwen_0
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/qwen_1
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/qwen_05
# python evaluation/story_quality_eval.py --run_id=cs4_tmp_var/qwen_025

######### olmo sft experiments #########
python evaluation/story_quality_eval.py --run_id=cs4_v1/olmo_13b_dpo
python evaluation/story_quality_eval.py --run_id=cs4_v1/olmo_13b_sft
python evaluation/constraint_satisfaction.py --run_id=cs4_v1/olmo_13b_dpo
python evaluation/constraint_satisfaction.py --run_id=cs4_v1/olmo_13b_sft
