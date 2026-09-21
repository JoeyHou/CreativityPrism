#!/usr/bin/env bash
# Phase 1 smoke tests for the unified runner.
# Run from repo root: bash task_runner/tests/test_phase1.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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
python task_runner/run.py --list-tasks
report "list-tasks" $?

echo
echo "=== 2. Listing models ==="
python task_runner/run.py --list-models > /dev/null
report "list-models" $?

echo
echo "=== 3. Dry-run for each task (no env needed) ==="
for t in aut ttcw creative_short ttct; do
    python task_runner/run.py --task "$t" --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run > /dev/null
    report "dry-run $t" $?
done

echo
echo "=== 4. Limit validation and forwarding ==="
for t in aut ttcw creative_short ttct; do
    output="$(python task_runner/run.py --task "$t" --model GPT4.1 --judge-model GPT4.1-mini --label smoke --limit 1 --dry-run 2>/dev/null)"
    rc=$?
    if [[ "$rc" -eq 0 && "$output" == *"--limit 1"* ]]; then
        report "limit forwarded for $t" 0
    else
        report "limit forwarded for $t" 1
    fi
done

output="$(python task_runner/run.py --config task_runner/fixtures/aut_smoke_v2.yaml --dry-run 2>/dev/null)"
rc=$?
if [[ "$rc" -eq 0 && "$output" == *"--limit 3"* ]]; then
    report "config limit forwarded" 0
else
    report "config limit forwarded" 1
fi

for invalid_limit in 0 -1; do
    python task_runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label smoke --limit "$invalid_limit" --dry-run >/dev/null 2>&1
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
        report "limit $invalid_limit rejected" 0
    else
        report "limit $invalid_limit rejected" 1
    fi
done

echo
echo "=== 5. Pre-flight: fails loudly when env is missing ==="
# Non-dry-run triggers preflight, which must fail *before* the adapter is invoked.
# CREATIVITYPRISM_ENV_PREFIX must point at a directory that cannot exist: without it
# this check runs a real, full-dataset, paid inference job on any machine where the
# env is actually built (i.e. every cluster), which is how it behaved until 2026-08-04.
CREATIVITYPRISM_ENV_PREFIX=/tmp/creativityprism-missing \
python task_runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label smoke 2>&1 | grep -q "registry/environments/setup.sh"
report "preflight error points to the env setup script" $?

echo
echo "=== 6. Error handling: unknown task ==="
python task_runner/run.py --task fake_task --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run 2>/dev/null
rc=$?
[[ "$rc" -ne 0 ]]
report "unknown task rejected" $?

echo
echo "=== 7. Error handling: unknown model ==="
python task_runner/run.py --task aut --model FakeModel --judge-model GPT4.1-mini --label smoke --dry-run 2>/dev/null
rc=$?
[[ "$rc" -ne 0 ]]
report "unknown model rejected" $?

echo
echo "=== 8. cs4 cleanup verification ==="
# 2026-08-05: the three original targets had already been deleted, so this
# check silently passed when grep found zero files. Now greps the whole repo.
# Excludes .ipynb: the old visualization notebook labels old cs4 columns, but
# those are display strings in a reference artifact, not functional references.
# Also excludes this file itself, since the search string appears in the grep
# command and would produce a self-hit that has nothing to do with cs4 cleanup.
# --exclude-dir=data: the check is about code and config, and the corpora under
# tasks/*/data are tens of MB on a network filesystem -- including them took the
# gate from 0.1s to over seven minutes once tasks/prism_v1/ landed.
cs4_hits="$(grep -rE "cs4|CS4" tasks/ registry/ scripts/ \
  --exclude=*.ipynb --exclude=README.md --exclude-dir=__pycache__ --exclude-dir=data 2>/dev/null || true)"
if [[ -z "$cs4_hits" ]]; then
    report "cs4 cleanup" 0
else
    echo "$cs4_hits"
    report "cs4 cleanup" 1
fi

echo
echo "=== 9. behavior regressions ==="
python -m unittest task_runner.tests.test_phase1_behavior
report "Phase 1 behavior regressions" $?

echo
echo "=== 9b. activate_env: conda envs must not take the venv branch ==="
# A conda env has bin/python too, so detecting a venv by that alone made
# activate_env return before exporting HF_HOME/TRITON_CACHE_DIR/TORCH_HOME.
# The Phase 1 "caches must not land in the quota-limited home" fix was inert
# from the day the venv branch landed until an open-weight run filled /ihome.
conda_env_exports="$(bash -c '
  cd "'"$REPO_ROOT"'"
  source registry/adapters/_common.sh 2>/dev/null
  activate_env modern >/dev/null 2>&1
  echo "${HF_HOME:-}|${TRITON_CACHE_DIR:-}|${TORCH_HOME:-}"
' 2>/dev/null)"
if [[ "$conda_env_exports" == *"|"*"|"* && "$conda_env_exports" != "||" ]]; then
    report "conda env gets cache redirection exports" 0
else
    echo "    got: '$conda_env_exports'"
    report "conda env gets cache redirection exports" 1
fi

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
