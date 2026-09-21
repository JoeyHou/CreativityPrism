#!/usr/bin/env bash
# model_hub gate. Deterministic: no network, no credentials, no GPU.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PASS=0; FAIL=0
report() {
    if [[ "$2" -eq 0 ]]; then echo "  PASS: $1"; PASS=$((PASS+1))
    else echo "  FAIL: $1 (rc=$2)"; FAIL=$((FAIL+1)); fi
}

echo "=== 1. unit tests ==="
python -m unittest task_runner.tests.test_model_hub 2>&1 | tail -3
report "model_hub unit tests" "${PIPESTATUS[0]}"

echo
echo "=== 2. imports without vLLM, torch or a GPU ==="
python -c "import sys; sys.path.insert(0,'.'); import model_hub; \
assert 'vllm' not in sys.modules and 'torch' not in sys.modules" 2>/dev/null
report "model_hub imports clean" $?

echo
echo "=== 3. provider modules import lazily (no SDK needed until used) ==="
python -c "
import sys; sys.path.insert(0,'.')
import model_hub.providers.openai_compat, model_hub.providers.anthropic, model_hub.providers.google
assert 'openai' not in sys.modules, 'openai imported at module load'
" 2>/dev/null
report "provider SDK imports are lazy" $?

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -eq 0 ]]
