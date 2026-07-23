#!/usr/bin/env bash
# Adapter for the AUT (Alternative Uses Task).
# Wraps tasks/aut_ttcw_cshort/run_inference.py with a one-shot ephemeral config.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

ALIAS="$(lookup_alias "$MODEL" aut_ttcw_cshort)"
API_ID="$(lookup_alias "$MODEL" api_call 2>/dev/null || echo "$ALIAS")"

TASK_DIR="$REPO_ROOT/tasks/aut_ttcw_cshort"
TMP_CONFIG="$(mktemp -t aut_cfg_XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG"' EXIT

TEST_SIZE_LINE=""
[[ -n "$LIMIT" ]] && TEST_SIZE_LINE='"test_size": '"$LIMIT"','

cat > "$TMP_CONFIG" <<JSON
{
    "experiments_list": [
        {
            ${TEST_SIZE_LINE}
            "run_id": "${RUN_ID}/aut/${ALIAS}",
            "task": "aut_push",
            "model_name": "${API_ID}"
        }
    ]
}
JSON

cd "$TASK_DIR"
if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    python run_inference.py "$TMP_CONFIG"
fi
# Eval is Phase 2; intentionally not implemented yet.

NATIVE_OUT="$TASK_DIR/data/output/${RUN_ID}/aut/${ALIAS}"
echo "OUTPUT_PATH=${NATIVE_OUT}"