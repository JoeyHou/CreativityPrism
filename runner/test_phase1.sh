#!/usr/bin/env bash
# Phase 1 smoke tests for the unified runner.
# Run from repo root: bash runner/test_phase1.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
report() {
    local name="$1"; local rc="$2"
    if [[ "$rc" -eq 0 ]]; then
        echo "  PASS: $name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $name (rc=$rc)"; FAIL=$((FAIL+1))
    fi
}

echo "=== 1. Listing tasks ==="
python runner/run.py --list-tasks
report "list-tasks" $?

echo
echo "=== 2. Listing models ==="
python runner/run.py --list-models > /dev/null
report "list-models" $?

echo
echo "=== 3. Dry-run for each task (no env needed) ==="
for t in aut ttcw creative_short ttct; do
    python runner/run.py --task "$t" --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run > /dev/null
    report "dry-run $t" $?
done

echo
echo "=== 4. Limit validation and forwarding ==="
for t in aut ttcw creative_short ttct; do
    output="$(python runner/run.py --task "$t" --model GPT4.1 --judge-model GPT4.1-mini --label smoke --limit 1 --dry-run 2>/dev/null)"
    rc=$?
    if [[ "$rc" -eq 0 && "$output" == *"--limit 1"* ]]; then
        report "limit forwarded for $t" 0
    else
        report "limit forwarded for $t" 1
    fi
done

output="$(python runner/run.py --config runs/aut_smoke_v2.yaml --dry-run 2>/dev/null)"
rc=$?
if [[ "$rc" -eq 0 && "$output" == *"--limit 3"* ]]; then
    report "config limit forwarded" 0
else
    report "config limit forwarded" 1
fi

for invalid_limit in 0 -1; do
    python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label smoke --limit "$invalid_limit" --dry-run >/dev/null 2>&1
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
        report "limit $invalid_limit rejected" 0
    else
        report "limit $invalid_limit rejected" 1
    fi
done

echo
echo "=== 5. Pre-flight: fails loudly when env is missing ==="
# Non-dry-run triggers preflight; since env doesn't exist yet, should print setup instructions.
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label smoke 2>&1 | grep -q "setup_envs"
report "preflight error points to setup_envs" $?

echo
echo "=== 6. Error handling: unknown task ==="
python runner/run.py --task fake_task --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run 2>/dev/null
rc=$?
[[ "$rc" -ne 0 ]]
report "unknown task rejected" $?

echo
echo "=== 7. Error handling: unknown model ==="
python runner/run.py --task aut --model FakeModel --judge-model GPT4.1-mini --label smoke --dry-run 2>/dev/null
rc=$?
[[ "$rc" -ne 0 ]]
report "unknown model rejected" $?

echo
echo "=== 8. cs4 cleanup verification ==="
if grep -rE "cs4|CS4" generate.py main/tasks/__init__.py tasks/aut_ttcw_cshort/scripts/ 2>/dev/null; then
    report "cs4 cleanup" 1
else
    report "cs4 cleanup" 0
fi

echo
echo "=== 9. behavior regressions ==="
python -m unittest runner.test_phase1_behavior
report "Phase 1 behavior regressions" $?

echo
echo "=== 10. registry layout sanity ==="
test -f registry/models.yaml \
  && test -f registry/tasks/aut.yaml \
  && test -f registry/tasks/ttcw.yaml \
  && test -f registry/tasks/creative_short.yaml \
  && test -f registry/tasks/ttct.yaml \
  && test -x registry/adapters/aut.sh \
  && test -x registry/adapters/ttcw.sh \
  && test -x registry/adapters/creative_short.sh \
  && test -x registry/adapters/ttct.sh \
  && test -f registry/environments/modern.txt
report "registry files present" $?

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -eq 0 ]]
