## Creative Math 
#### Installing Dependencies:
pip install vllm==0.7.3

#### Inference
First, please create a config file at the configs directory. We offered an example of how to load the existing models.

Next, please add the new model into the inference_driver in either the closed_source_model or the open_source_models.

You can do inference with python run_inference_all.py configs/inference_creative_math.json 0

Currently, we hard coded the hyperparameters while performing inference to ensure the correctness of the parameters. Nevertheless, the code also support for using the config files. 

#### Data Cleaning
After running the inference, the data may contain information that are distractive to judge models such as explaining why the results are novel. We remove such distractors by using llama 3.3 70B Instruct to extract the solution out from the model's raw responses. An example is provided below for reference.

python -m src.utils.clean_data_creative_math --input_file "data/outputs/creative_math/OLMo2-13B-sft/OLMo2-13B-sft.json" --output_file "data/outputs/creative_math_filtered_temp0.1_extract/OLMo2-13B-sft.json"

#### Evaluation
An example of how to run the evaluation is provided below. Please be careful with the configs and data path you are using to ensure the evaluation is carried out on the cleaned model outputs.

python -m src.evaluation.creative_math_eval_api --model_to_evaluate model_name --portion portion_of_data_to_use for evaluation for api-based models


Please note that you need to input your api keys in the inference_driver.py and src.evaluation.creative_math_eval_api.py for the process to start. In evaluation, only Claude is used as the correctness evaluator; all other three evaluators are used for the coarsed-grained novelty evaluation. Please comment out those not useful evaluators. We will keep revising the script for easier usage.


## Creativity Index
#### Installing Dependencies:
pip install vllm==0.7.3
Note that we are using request for the inifi-gram api, so it is not necessary to use infini-gram locally at here

#### Inference 
Below is an example usage for running inference with creative index

Please make sure to create your config file in the format of the provided example. Please specify which task to use in the config file.

python run_inference_all.py configs/inference_creative_index_book.json 0


#### Evaluation
Example for the evaluation process:
    python -m src.evaluation.evaluation_creative_index_parr \
        --task OLMo2-13B-sft-book-dolma \
        --data data/outputs/new_index/OLMo2-13B-sft/book.json \
        --output_dir data/OLMo2-13B-sft \
        --min_ngram $MIN_NGRAM \
        --subset 100 \
        --lm_tokenizer \
        --num_workers 8

The --subset refers to how many data you want to evaluate on. The --lm_tokenizer field is whether to use llama 2 tokenizer for parsing the output. In our evaluation, we used this tokenizer for parsing.


