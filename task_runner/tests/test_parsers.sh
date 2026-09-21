#!/usr/bin/env bash
# Gate for the judge-verdict parsers inside tasks/prism_v1.
#
# Separate from the other gates because it is the only one that imports task
# code rather than runner code. It needs the task env (openai, pandas) but no
# network, no GPU, and no API key.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/tasks/prism_v1" || exit 1

PASS=0; FAIL=0
echo "== judge-verdict parsers =="
if python -m unittest tests.test_parsers 2>&1 | tail -3 | grep -q "^OK"; then
    echo "  PASS: verdict parsers"
    PASS=$((PASS+1))
else
    echo "  FAIL: verdict parsers"
    python -m unittest tests.test_parsers 2>&1 | tail -20
    FAIL=$((FAIL+1))
fi

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
