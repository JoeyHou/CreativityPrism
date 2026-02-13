# done
python run_inference.py configs/aut/inference/olmo_sft.json
python run_inference.py configs/ttcw/inference/olmo_sft.json
python run_inference.py configs/creative_short/olmo_sft.json

# done
cd ../cs4_benchmark/story_generator
python cs4_story_generator.py configs/olmo_sft.json
