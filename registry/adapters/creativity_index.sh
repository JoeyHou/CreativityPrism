#!/usr/bin/env bash
# Adapter for Creativity Index (book / poem / speech continuations scored by
# n-gram overlap against a web-scale corpus).
#
# --judge-model is a documented NO-OP here: the metric is exact n-gram matching
# via the infini-gram API, not an LLM judge.
#
# Network: evaluation queries https://api.infini-gram.io/ and downloads a
# Llama-2-7b tokenizer, so eval mode needs outbound internet from the compute node.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern
export_provider_keys

ALIAS="$(lookup_alias "$MODEL" math_n_index)"

TASK_DIR="$REPO_ROOT/tasks/math_n_index"
cd "$TASK_DIR"
add_pythonpath "$(pwd)"

# One registry task covers all three domains, matching how the benchmark reports
# Creativity Index. Override to run a subset, e.g. CREATIVITYPRISM_INDEX_DOMAINS="poem".
read -r -a DOMAINS <<< "${CREATIVITYPRISM_INDEX_DOMAINS:-book poem speech}"
# The bundled scripts (book_creative_par.sh and friends) sweep 5..12; keep that
# default so numbers stay comparable, but allow a cheaper single-n run.
read -r -a MIN_NGRAMS <<< "${CREATIVITYPRISM_INDEX_MIN_NGRAM:-5 6 7 8 9 10 11 12}"
NUM_WORKERS="${CREATIVITYPRISM_INDEX_WORKERS:-8}"
MAX_NEW_TOKENS="${CREATIVITYPRISM_INDEX_MAX_NEW_TOKENS:-288}"
TEMP="${CREATIVITYPRISM_INDEX_TEMP:-0.75}"
TOP_P="${CREATIVITYPRISM_INDEX_TOP_P:-0.9}"

GEN_DIR="data/outputs/${RUN_ID}/creative_index/${ALIAS}"
EVAL_DIR="data/evaluations/${RUN_ID}/creative_index/${ALIAS}"

CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$CONFIG_DIR"' EXIT

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    mkdir -p "$GEN_DIR"
    for domain in "${DOMAINS[@]}"; do
        INFER_CONFIG="$CONFIG_DIR/inference_creative_index_${domain}.json"
        CP_ALIAS="$ALIAS" CP_DOMAIN="$domain" CP_GEN="$GEN_DIR" \
        CP_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" CP_TEMP="$TEMP" CP_TOP_P="$TOP_P" \
        CP_TEST_SIZE="${LIMIT:--1}" CP_OUT="$INFER_CONFIG" python3 <<'PY'
import json
import os

domain = os.environ["CP_DOMAIN"]
config = {
    "experiments_list": [
        {
            "task": domain,
            "model_name": os.environ["CP_ALIAS"],
            "portion": 1.0,
            "test_size": int(os.environ["CP_TEST_SIZE"]),
            "model_config": {
                "max_new_tokens": int(os.environ["CP_MAX_NEW_TOKENS"]),
                "temperature": float(os.environ["CP_TEMP"]),
                "top_p": float(os.environ["CP_TOP_P"]),
                "seed": 42,
            },
            "file_paths": {
                "dataset": f"data/processed/creative_index/{domain}_prompt.json",
                "generation": os.environ["CP_GEN"],
            },
        }
    ]
}
with open(os.environ["CP_OUT"], "w") as handle:
    json.dump(config, handle, indent=2)
PY
        python3 run_inference_all.py "$INFER_CONFIG" 0
    done
    # One artifact for the whole task: the run-scoped directory holding
    # book.json / poem.json / speech.json.
    emit_artifact inference "$TASK_DIR/${GEN_DIR}"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    mkdir -p "$EVAL_DIR"
    for domain in "${DOMAINS[@]}"; do
        GEN_FILE="${GEN_DIR}/${domain}.json"
        # --subset defaults to 100 and silently truncates, so always pass it
        # explicitly: the requested limit, or the full generation file.
        if [[ -n "$LIMIT" ]]; then
            SUBSET="$LIMIT"
        else
            SUBSET="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$GEN_FILE")"
        fi
        for min_ngram in "${MIN_NGRAMS[@]}"; do
            python3 -m src.evaluation.evaluation_creative_index_parr \
                --task "${ALIAS}_${domain}" \
                --data "$GEN_FILE" \
                --output_dir "$EVAL_DIR" \
                --min_ngram "$min_ngram" \
                --subset "$SUBSET" \
                --lm_tokenizer \
                --num_workers "$NUM_WORKERS"
        done
    done
    emit_artifact eval "$TASK_DIR/${EVAL_DIR}"
fi
