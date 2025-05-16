# good - CS4 sft inf
python cs4_story_generator.py configs/olmo_sft.json

# good - CS4 tmp_var inf
python cs4_story_generator.py configs/tmp_var/inference_olmo_0.json
python cs4_story_generator.py configs/tmp_var/inference_olmo_1.json
python cs4_story_generator.py configs/tmp_var/inference_olmo_05.json
python cs4_story_generator.py configs/tmp_var/inference_olmo_025.json

# change dir
cd ../../creative_bench/

# good - AUT/TTCT/CreativeShort sft inf
python run_inference.py configs/aut/inference/olmo_sft.json
python run_inference.py configs/ttcw/inference/olmo_sft.json
python run_inference.py configs/creative_short/olmo_sft.json

# good - AUT tmp_var inf
python run_inference.py configs/aut/tmp_var/inference_olmo_0.json
python run_inference.py configs/aut/tmp_var/inference_olmo_1.json
python run_inference.py configs/aut/tmp_var/inference_olmo_05.json
python run_inference.py configs/aut/tmp_var/inference_olmo_025.json
# python run_inference.py configs/aut/tmp_var/inference_qwen_0.json
# python run_inference.py configs/aut/tmp_var/inference_qwen_1.json
# python run_inference.py configs/aut/tmp_var/inference_qwen_05.json
# python run_inference.py configs/aut/tmp_var/inference_qwen_025.json

# good - TTCW tmp_var inf
python run_inference.py configs/ttcw/tmp_var/inference_olmo_0.json
python run_inference.py configs/ttcw/tmp_var/inference_olmo_1.json
python run_inference.py configs/ttcw/tmp_var/inference_olmo_05.json
python run_inference.py configs/ttcw/tmp_var/inference_olmo_025.json
# python run_inference.py configs/ttcw/tmp_var/inference_qwen_0.json
# python run_inference.py configs/ttcw/tmp_var/inference_qwen_1.json
# python run_inference.py configs/ttcw/tmp_var/inference_qwen_05.json
# python run_inference.py configs/ttcw/tmp_var/inference_qwen_025.json

# good - CreativeShort tmp_var inf
python run_inference.py configs/creative_short/tmp_var/inference_olmo_0.json
python run_inference.py configs/creative_short/tmp_var/inference_olmo_1.json
python run_inference.py configs/creative_short/tmp_var/inference_olmo_05.json
python run_inference.py configs/creative_short/tmp_var/inference_olmo_025.json
# python run_inference.py configs/creative_short/tmp_var/inference_qwen_0.json
# python run_inference.py configs/creative_short/tmp_var/inference_qwen_1.json
# python run_inference.py configs/creative_short/tmp_var/inference_qwen_05.json
# python run_inference.py configs/creative_short/tmp_var/inference_qwen_025.json

