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

### Prerequisite: GloVe vectors

Scoring needs `glove.840B.300d.txt` (~2 GB), which is **not** shipped with this repo:

```bash
mkdir -p embeddings/glove
curl -L -o /tmp/glove.zip https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip /tmp/glove.zip -d embeddings/glove
```

`embeddings/` is gitignored. If you already have a copy elsewhere, point
`--glove-path` or `$CREATIVITYPRISM_GLOVE_PATH` at it instead. Inference does not
need the file; only `steps/evaluate_dat.py` does, and it fails with this
instruction rather than a bare traceback when the file is missing.

The word list defaults to the bundled `words.txt`; override with `--words-path`
or `$CREATIVITYPRISM_DAT_WORDS`.

### Inference
See a running example in `script/inference_dat.sh`

### Evaluation
See a running example in `script/evaluate_dat.sh`

Pass `--output-path` explicitly. The historical default rewrites `inference` to
`evaluation` inside the input path, which silently overwrites the inference file
when the path contains no such segment.