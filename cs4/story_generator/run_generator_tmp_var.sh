# good - moved to temp_var_inference_small.sh
python cs4_story_generator.py configs/olmo_sft.json

# good - moved to temp_var_inference_small.sh
python cs4_story_generator.py configs/tmp_var/inference_olmo_0.json
python cs4_story_generator.py configs/tmp_var/inference_olmo_1.json
python cs4_story_generator.py configs/tmp_var/inference_olmo_05.json
python cs4_story_generator.py configs/tmp_var/inference_olmo_025.json

# unchecked
# python cs4_story_generator.py configs/tmp_var/inference_qwen_0.json
# python cs4_story_generator.py configs/tmp_var/inference_qwen_1.json
# python cs4_story_generator.py configs/tmp_var/inference_qwen_05.json
# python cs4_story_generator.py configs/tmp_var/inference_qwen_025.json