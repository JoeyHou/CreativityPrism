Please check https://github.com/JHU-CLSP/NeoCoder and https://github.com/DingNLab/probing_creativity for environment setup.

# NeoCoder

### Inference (Equation 1 in the paper)
1. Inference on NeoCoder dataset: `python steps/inference_dp.py --dataset-path --model-name {HF_MODEL_NAME, OPENAI_MODEL_NAME} --dp-rounds --batch-size --output-dir`

   We provide a running example in `scripts/inference_neocoder`

### NeoGauge@T Calculation (Section 4 in the paper)
1. Detect Techniques: `python steps/evaluate_neogauge.py --task detection --inference-result-path --human-solution-path`

   We provide a running example in `scripts/detect_techniques.sh`

2. Evaluate correctness: `python steps/evaluate_neogauge.py --task correctness --inference-result-path --test-case-path --save-folder --model-family`

   We provide a running example in `scripts/correctness_evaluation.sh`

3. Final NeoGauge@T Calculation: `python steps/evaluate_neogauge.py --task creativity --inference-result-path --human-solution-path --save-folder`
   
   We provide a running example in `scripts/evaluate_neogauge.sh`

# DAT

### Inference
See a running example in `script/inference_dat.sh`

### Evaluation
See a running example in `script/evaluate_dat.sh`