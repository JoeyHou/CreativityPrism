#!/usr/bin/env bash
# Gate for the pre-flight checker (model_hub/smoke.py).
# Run from repo root: bash task_runner/tests/test_smoke.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
report() {
    if [[ "$2" -eq 0 ]]; then echo "  PASS: $1"; PASS=$((PASS+1))
    else echo "  FAIL: $1 (rc=$2)"; FAIL=$((FAIL+1)); fi
}

echo "=== 1. classification unit tests (no network) ==="
python -m unittest task_runner.tests.test_smoke -v 2>&1 | tail -3
python -m unittest task_runner.tests.test_smoke >/dev/null 2>&1
report "smoke classification" $?

echo
echo "=== 2. --list works without credentials ==="
CREATIVITYPRISM_API_KEYS=/tmp/creativityprism-no-such-keys.json \
  python -m model_hub.smoke --list 2>/dev/null | grep -q "GPT4.1"
report "--list needs no credentials" $?

echo
echo "=== 3. no arguments is an error, not a silent pass ==="
python -m model_hub.smoke >/dev/null 2>&1
[[ $? -ne 0 ]]
report "bare invocation rejected" $?

echo
echo "=== 4. an unusable model makes the exit code non-zero ==="
# The whole point is that a run script can gate on this.
python -m model_hub.smoke --models DefinitelyNotARegisteredModel >/dev/null 2>&1
[[ $? -ne 0 ]]
report "failure sets exit code" $?

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -eq 0 ]]
