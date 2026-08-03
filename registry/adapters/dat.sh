#!/usr/bin/env bash
# Adapter for DAT (Divergent Association Task).
#
# --judge-model is a documented NO-OP here: scoring is mean pairwise GloVe distance,
# not an LLM judge.
#
# --limit maps to --repeat. DAT has a single fixed prompt rather than a dataset, so
# the number of repeats *is* the sample count.
#
# PREREQUISITE: evaluation needs glove.840B.300d.txt (~2 GB), which is not shipped
# with this repo. See tasks/neocoder_dat/README.md. Inference does not need it.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env legacy
export_provider_keys

ALIAS="$(lookup_alias "$MODEL" neocoder_dat)"

TASK_DIR="$REPO_ROOT/tasks/neocoder_dat"
cd "$TASK_DIR"
add_pythonpath "$(pwd)"

REPEAT="${LIMIT:-${CREATIVITYPRISM_DAT_REPEAT:-100}}"
TEMP="${CREATIVITYPRISM_DAT_TEMP:-0.75}"
TOP_P="${CREATIVITYPRISM_DAT_TOP_P:-1}"
MAX_TOKENS="${CREATIVITYPRISM_DAT_MAX_TOKENS:-4096}"

# Run isolation: --output-dir is required, so scoping it by <run_id> is enough.
INFER_DIR="results/${RUN_ID}/dat/inference"
EVAL_DIR="results/${RUN_ID}/dat/evaluation"

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    mkdir -p "$INFER_DIR"
    python3 steps/inference_dat.py \
        --model-name "$ALIAS" \
        --repeat "$REPEAT" \
        --output-dir "$INFER_DIR" \
        --overwrite \
        --temperature "$TEMP" \
        --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS"
    # The filename embeds int(temperature * 100), which floating point makes
    # awkward to reproduce here; the directory is run-scoped, so glob it instead.
    emit_artifact inference "$TASK_DIR/${INFER_DIR}"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    shopt -s nullglob
    INFER_FILES=("${INFER_DIR}"/*.json)
    shopt -u nullglob
    if [[ ${#INFER_FILES[@]} -ne 1 ]]; then
        echo "dat.sh: expected exactly 1 inference file in ${INFER_DIR}, found ${#INFER_FILES[@]}" >&2
        exit 1
    fi
    INFER_FILE="${INFER_FILES[0]}"
    EVAL_FILE="${EVAL_DIR}/$(basename "$INFER_FILE")"

    mkdir -p "$EVAL_DIR"
    # --output-path is passed explicitly: the script's default rewrites "inference"
    # to "evaluation" inside the input path, which is fragile.
    python3 steps/evaluate_dat.py \
        --result-path "$INFER_FILE" \
        --output-path "$EVAL_FILE"
    emit_artifact eval "$TASK_DIR/${EVAL_FILE}"
fi
