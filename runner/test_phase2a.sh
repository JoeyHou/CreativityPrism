#!/usr/bin/env bash
# Phase 2A gate: artifact contract, centralized outputs, metadata.
# Run from repo root: bash runner/test_phase2a.sh
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

echo "=== 1. artifact contract regressions ==="
python -m unittest runner.test_phase2a_artifacts
report "artifact contract regressions" $?

echo
echo "=== 2. dry-run creates no outputs/ side effects ==="
BEFORE="absent"; [[ -e outputs ]] && BEFORE="present"
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini \
    --label phase2a_dryrun --dry-run > /dev/null 2>&1
AFTER="absent"; [[ -e outputs ]] && AFTER="present"
if [[ "$BEFORE" == "$AFTER" && ! -e outputs/phase2a_dryrun ]]; then
    report "dry-run has no side effects" 0
else
    report "dry-run has no side effects" 1
fi

echo
echo "=== 3. runner announces the centralized output dir ==="
output="$(python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini \
    --label phase2a_dryrun --dry-run 2>/dev/null)"
if [[ "$output" == *"--output-dir"*"outputs/phase2a_dryrun/aut/GPT4.1"* ]] \
   || [[ "$output" == *"--output-dir"*"outputs\\phase2a_dryrun\\aut\\GPT4.1"* ]]; then
    report "adapter receives centralized --output-dir" 0
else
    report "adapter receives centralized --output-dir" 1
fi

echo
echo "=== 4. outputs/ is git-ignored ==="
git check-ignore -q outputs/probe/aut/GPT4.1/metadata.json
report "outputs/ ignored by git" $?

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -eq 0 ]]
