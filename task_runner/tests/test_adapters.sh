#!/usr/bin/env bash
# Gate: all eight adapters emit configs tasks/prism_v1 can dispatch.
# No GPU, no API call. Needs PyYAML (the modern env has it).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

PASS=0; FAIL=0
echo "== adapter configs =="
OUT="$(python -m unittest task_runner.tests.test_adapters 2>&1)"
if echo "$OUT" | tail -3 | grep -q "^OK"; then
    echo "  PASS: 8 adapters x 2 stages dispatchable"
    PASS=$((PASS+1))
else
    echo "  FAIL: adapter configs"
    echo "$OUT" | tail -25
    FAIL=$((FAIL+1))
fi

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
