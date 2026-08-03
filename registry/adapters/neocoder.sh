#!/usr/bin/env bash
# Adapter for NeoCoder (Codeforces problems under denial-of-prompt constraints).
#
# --judge-model is a documented NO-OP here: the technique-detection stage hardcodes
# gpt-4-turbo as the code reviewer inside src/utils/configs.py.
#
# SAFETY: the correctness stage executes model-generated Python. That is inherent
# to the benchmark; run it only on a machine where that is acceptable.
#
# COST: technique detection issues one gpt-4-turbo call per generated solution.
#
# Eval is a strict three-step chain because calculate_creativity() asserts that
# both "correctness" and "techniques" have been written back into the inference
# file: correctness -> detection -> creativity.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env legacy
export_provider_keys

ALIAS="$(lookup_alias "$MODEL" neocoder_dat)"
MODEL_SHORT="${ALIAS##*/}"

TASK_DIR="$REPO_ROOT/tasks/neocoder_dat"
cd "$TASK_DIR"
add_pythonpath "$(pwd)"

DATASET="datasets/CodeForce/NeoCoder/NeoCoder.json"
HUMAN_SOLUTIONS="datasets/CodeForce/NeoCoder/human_solutions.json"
TEST_CASES="datasets/CodeForce/NeoCoder/test_cases_annotated.json"

# 5 matches scripts/inference_neocoder.sh and the dp_rounds default baked into
# calculate_creativity(); changing one without the other skews the score.
DP_ROUNDS="${CREATIVITYPRISM_NEOCODER_DP_ROUNDS:-5}"
BATCH_SIZE="${CREATIVITYPRISM_NEOCODER_BATCH_SIZE:-1}"
TEMP="${CREATIVITYPRISM_NEOCODER_TEMP:-0.75}"
TOP_P="${CREATIVITYPRISM_NEOCODER_TOP_P:-1}"
MAX_TOKENS="${CREATIVITYPRISM_NEOCODER_MAX_TOKENS:-1024}"

# Run isolation: --output-dir / --save-folder are required arguments, so scoping
# them by <run_id> is enough. correctness and creativity get separate folders
# because configs.py writes both under the same "<model>_sample=..._creativity.json"
# filename and would otherwise clobber each other.
INFER_DIR="results/${RUN_ID}/neocoder/inference"
CORRECTNESS_DIR="results/${RUN_ID}/neocoder/evaluation/correctness"
CREATIVITY_DIR="results/${RUN_ID}/neocoder/evaluation/creativity"

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    mkdir -p "$INFER_DIR"
    LIMIT_ARG=()
    [[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")
    python3 steps/inference_dp.py \
        --dataset-path "$DATASET" \
        --model-name "$ALIAS" \
        --dp-rounds "$DP_ROUNDS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$INFER_DIR" \
        --overwrite \
        --temperature "$TEMP" \
        --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS" \
        "${LIMIT_ARG[@]}"
    # The filename embeds sample count and temperature, which only the task can
    # compute, so announce the run-scoped directory instead of guessing.
    emit_artifact inference "$TASK_DIR/${INFER_DIR}"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    # Exactly one inference file exists per run because INFER_DIR is run-scoped.
    shopt -s nullglob
    INFER_FILES=("${INFER_DIR}"/*.json)
    shopt -u nullglob
    if [[ ${#INFER_FILES[@]} -ne 1 ]]; then
        echo "neocoder.sh: expected exactly 1 inference file in ${INFER_DIR}, found ${#INFER_FILES[@]}" >&2
        exit 1
    fi
    INFER_FILE="${INFER_FILES[0]}"

    mkdir -p "$CORRECTNESS_DIR" "$CREATIVITY_DIR"

    # Detection runs first because it writes `techniques` back into the inference file;
    # correctness then carries that field into its own output, which is the only file
    # holding both of the fields calculate_creativity asserts on.
    python3 steps/evaluate_neogauge.py \
        --task detection \
        --inference-result-path "$INFER_FILE" \
        --human-solution-path "$HUMAN_SOLUTIONS"

    python3 steps/evaluate_neogauge.py \
        --task correctness \
        --inference-result-path "$INFER_FILE" \
        --test-case-path "$TEST_CASES" \
        --save-folder "$CORRECTNESS_DIR"

    shopt -s nullglob
    SCORED_FILES=("${CORRECTNESS_DIR}"/*.json)
    shopt -u nullglob
    if [[ ${#SCORED_FILES[@]} -ne 1 ]]; then
        echo "neocoder.sh: expected exactly 1 correctness file in ${CORRECTNESS_DIR}, found ${#SCORED_FILES[@]}" >&2
        exit 1
    fi

    python3 steps/evaluate_neogauge.py \
        --task creativity \
        --inference-result-path "${SCORED_FILES[0]}" \
        --human-solution-path "$HUMAN_SOLUTIONS" \
        --save-folder "$CREATIVITY_DIR"

    emit_artifact eval "$TASK_DIR/${CREATIVITY_DIR}"
fi
