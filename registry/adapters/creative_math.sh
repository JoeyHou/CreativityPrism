#!/usr/bin/env bash
# Adapter for Creative Math (alternative-solution generation on math problems).
#
# --judge-model is a documented NO-OP here. src/evaluation/creative_math_eval_api.py
# scores every item with a fixed three-judge panel (gpt-4.1, claude-3-7-sonnet,
# gemini-2.0-flash) and takes a majority vote; there is no single-judge mode, and
# swapping one judge would not be comparable to the published numbers.
#
# The task is driven entirely by JSON configs, so run isolation needs no task-side
# patch: this adapter generates both configs under a temp dir and points every
# output path at <run_id>.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern
export_provider_keys

ALIAS="$(lookup_alias "$MODEL" math_n_index)"

TASK_DIR="$REPO_ROOT/tasks/math_n_index"
cd "$TASK_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATASET="data/processed/creative_math.json"
# creative_math_eval_api.py rebuilds the generation path as
#   <generation>/<alias>/<alias>.json
# so inference writes into GEN_DIR and eval is handed GEN_PARENT.
GEN_PARENT="data/outputs/${RUN_ID}/creative_math"
GEN_DIR="${GEN_PARENT}/${ALIAS}"
EVAL_DIR="data/evaluations/${RUN_ID}/creative_math"
LOG_DIR="logs/${RUN_ID}"

MAX_NEW_TOKENS="${CREATIVITYPRISM_MATH_MAX_NEW_TOKENS:-2000}"
TEMP="${CREATIVITYPRISM_MATH_TEMP:-0}"

CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$CONFIG_DIR"' EXIT

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    INFER_CONFIG="$CONFIG_DIR/inference_creative_math.json"
    # Values are passed through the environment, never interpolated into the
    # Python source, so an odd alias or path cannot break out of the heredoc.
    CP_ALIAS="$ALIAS" CP_DATASET="$DATASET" CP_GEN="$GEN_DIR" CP_EVAL="$EVAL_DIR" \
    CP_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" CP_TEMP="$TEMP" CP_TEST_SIZE="${LIMIT:--1}" \
    CP_OUT="$INFER_CONFIG" python3 <<'PY'
import json
import os

config = {
    "experiments_list": [
        {
            "task": "creative_math",
            "model_name": os.environ["CP_ALIAS"],
            "portion": 1.0,
            "test_size": int(os.environ["CP_TEST_SIZE"]),
            "model_config": {
                "max_new_tokens": int(os.environ["CP_MAX_NEW_TOKENS"]),
                "temperature": float(os.environ["CP_TEMP"]),
                "seed": 42,
            },
            "file_paths": {
                "dataset": os.environ["CP_DATASET"],
                "generation": os.environ["CP_GEN"],
                "evaluation": os.environ["CP_EVAL"],
            },
        }
    ]
}
with open(os.environ["CP_OUT"], "w") as handle:
    json.dump(config, handle, indent=2)
PY

    mkdir -p "$GEN_DIR"
    # argv[2] is the index into experiments_list; the generated config has exactly one.
    python3 run_inference_all.py "$INFER_CONFIG" 0
    emit_artifact inference "$TASK_DIR/${GEN_DIR}/${ALIAS}.json"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    # creative_math_eval_api.py reads a FLAT config (config["file_paths"], not
    # experiments_list) at import time from CREATIVITYPRISM_MATH_EVAL_CONFIG.
    EVAL_CONFIG="$CONFIG_DIR/eval_creative_math.json"
    CP_DATASET="$DATASET" CP_GEN_PARENT="$GEN_PARENT" CP_EVAL="$EVAL_DIR" \
    CP_LOG_DIR="$LOG_DIR" CP_OUT="$EVAL_CONFIG" python3 <<'PY'
import json
import os

config = {
    "file_paths": {
        "dataset": os.environ["CP_DATASET"],
        "generation": os.environ["CP_GEN_PARENT"],
        "evaluation": os.environ["CP_EVAL"],
    },
    "logging": {"log_level": "INFO", "log_dir": os.environ["CP_LOG_DIR"]},
}
with open(os.environ["CP_OUT"], "w") as handle:
    json.dump(config, handle, indent=2)
PY

    mkdir -p "$LOG_DIR" "$EVAL_DIR"
    CREATIVITYPRISM_MATH_EVAL_CONFIG="$EVAL_CONFIG" \
        python3 -m src.evaluation.creative_math_eval_api \
        --model_to_evaluate "$ALIAS" \
        --portion 1
    emit_artifact eval "$TASK_DIR/${EVAL_DIR}/${ALIAS}_Claude_correctness.json"
fi
