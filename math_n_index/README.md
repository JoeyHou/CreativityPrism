## Creative Math 
#### Installing Dependencies:
```bash
pip install vllm==0.7.3
```
#### Inference
First, please create a config file in the configs directory. We offered an example of how to load the existing models.

Next, please add the new model to the inference_driver in either the closed_source_model or the open_source_models.

You can do inference with 
```bash
Python run_inference_all.py configs/inference_creative_math.json 0
```
Currently, we have hard-coded the hyperparameters while performing inference and evaluation to ensure the correctness of the parameters. Nevertheless, the code also supports using the config files. 

#### Data Cleaning
After running the inference, the data may contain information that is distracting to judge models, such as explaining why the results are novel. We remove such distractors by using Llama 3.3 70 B. Instruct to extract the solution from the model's raw responses. An example is provided below for reference.

```bash
python -m src.utils.clean_data_creative_math --input_file "data/outputs/creative_math/OLMo2-13B-sft/OLMo2-13B-sft.json" --output_file "data/outputs/creative_math_filtered_temp0.1_extract/OLMo2-13B-sft.json"
```

#### Evaluation
An example of how to run the evaluation is provided below. Please be careful with the configs and data path you are using to ensure the evaluation is carried out on the cleaned model outputs. Please be careful when setting the generation path in the config file to the **cleaned** data, not the originally generated data. Please also adjust the number of tokens you want to use in evaluation in model_wrapper.py. If you intend to include the reasons why judge models reached the decision, please revise the prompt in src/prompt_engineering/creative_math_prompts.py. An example of how to use the evaluation code is provided below. 

```bash
python -m src.evaluation.creative_math_eval_api --model_to_evaluate model_name --portion portion_of_data_to_use for evaluation for api-based models
```

Please note that you need to input your api keys in the inference_driver.py and src.evaluation.creative_math_eval_api.py for the process to start. In evaluation, only Claude is used as the correctness evaluator; all three other evaluators are used for the coarse-grained novelty evaluation. Please comment out those not useful evaluators. We will keep revising the script for easier usage.

#### Output
The output will appear in the log information. You can direct the log to record them, or directly record the results at the end of the evaluation. Currently, we only consider the correctness ratio and coarse-grained novelty in evaluation. Please refer to the paper on why we exclude the fine-grained evaluation. 


## Creativity Index
#### Installing Dependencies:
```bash
pip install vllm==0.7.3
```
Note that we are using request for the inifi-gram api, so it is not necessary to use Inifi-gram locally here

#### Inference 
Below is an example usage for running inference with the Creative Index

Please make sure to create your config file in the format of the provided example. Please specify which task to use in the config file.
```bash
python run_inference_all.py configs/inference_creative_index_book.json 0
```

#### Evaluation
Example for the evaluation process:

```bash
python -m src.evaluation.evaluation_creative_index_parr \
        --task OLMo2-13B-sft-book-dolma \
        --data data/outputs/new_index/OLMo2-13B-sft/book.json \
        --output_dir data/OLMo2-13B-sft \
        --min_ngram $MIN_NGRAM \
        --subset 100 \
        --lm_tokenizer \
        --num_workers 8
```

The --subset refers to how much data you want to evaluate on. The --lm_tokenizer field is whether to use the llama 2 tokenizer for parsing the output. In our evaluation, we used this tokenizer for parsing. Please refer to the original repository of Creative Index for a more detailed environmental setup, or if you want to use the semantic match version of the Creative Index instead of the currently supported exact match method. Please refer to the paper for why we chose to only include the exact match metric. 

#### Output
The output will be stored in files with the following format:
```bash
args.output_dir/args.task'_exact_'str(args.min_ngram)'.json'
```
To aggregate the result and get the Creative Index, please loop through the JSON files and extract the L-uniquness for each data point with 1 - coverage. Please refer to the paper for how the Creative Index is calculated for more information.
