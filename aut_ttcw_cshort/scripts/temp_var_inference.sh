# CS4
python cs4_story_generator.py configs/tmp_var/inference_qwen_0.json
python cs4_story_generator.py configs/tmp_var/inference_qwen_1.json
python cs4_story_generator.py configs/tmp_var/inference_qwen_05.json
python cs4_story_generator.py configs/tmp_var/inference_qwen_025.json

# change dir
cd ../../creative_bench/

# good - AUT
# python run_inference.py configs/aut/tmp_var/inference_olmo_0.json
# python run_inference.py configs/aut/tmp_var/inference_olmo_1.json
# python run_inference.py configs/aut/tmp_var/inference_olmo_05.json
# python run_inference.py configs/aut/tmp_var/inference_olmo_025.json
python run_inference.py configs/aut/tmp_var/inference_qwen_0.json
python run_inference.py configs/aut/tmp_var/inference_qwen_1.json
python run_inference.py configs/aut/tmp_var/inference_qwen_05.json
python run_inference.py configs/aut/tmp_var/inference_qwen_025.json

# good - TTCW
# python run_inference.py configs/ttcw/tmp_var/inference_olmo_0.json
# python run_inference.py configs/ttcw/tmp_var/inference_olmo_1.json
# python run_inference.py configs/ttcw/tmp_var/inference_olmo_05.json
# python run_inference.py configs/ttcw/tmp_var/inference_olmo_025.json
python run_inference.py configs/ttcw/tmp_var/inference_qwen_0.json
python run_inference.py configs/ttcw/tmp_var/inference_qwen_1.json
python run_inference.py configs/ttcw/tmp_var/inference_qwen_05.json
python run_inference.py configs/ttcw/tmp_var/inference_qwen_025.json

# good - CreativeShort
# python run_inference.py configs/creative_short/tmp_var/inference_olmo_0.json
# python run_inference.py configs/creative_short/tmp_var/inference_olmo_1.json
# python run_inference.py configs/creative_short/tmp_var/inference_olmo_05.json
# python run_inference.py configs/creative_short/tmp_var/inference_olmo_025.json
python run_inference.py configs/creative_short/tmp_var/inference_qwen_0.json
python run_inference.py configs/creative_short/tmp_var/inference_qwen_1.json
python run_inference.py configs/creative_short/tmp_var/inference_qwen_05.json
python run_inference.py configs/creative_short/tmp_var/inference_qwen_025.json

