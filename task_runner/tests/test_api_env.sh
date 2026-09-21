#!/usr/bin/env bash
# API-env gate: every task module must import without vLLM installed.
# Run from repo root: bash task_runner/tests/test_api_env.sh
#
# Skips cleanly when the api env is absent, so it is safe to chain after the
# other gates on machines that only have the conda envs.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

prefix=""
if [[ -n "${CREATIVITYPRISM_ENV_PREFIX:-}" ]]; then
    prefix="$CREATIVITYPRISM_ENV_PREFIX"
elif [[ -f registry/environments/.location ]]; then
    prefix="$(cat registry/environments/.location)"
else
    prefix="$REPO_ROOT/registry/environments"
fi

VENV="${prefix%/}/creativityprism-api"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || PY="$VENV/Scripts/python.exe"
if [[ ! -x "$PY" ]]; then
    echo "SKIP: api env not found at $VENV (bash registry/environments/setup.sh --env api)"
    exit 0
fi

if "$PY" -c "import vllm" >/dev/null 2>&1; then
    echo "SKIP: vLLM is installed in $VENV, so this gate proves nothing"
    exit 0
fi

echo "=== task modules import without vLLM ($PY) ==="
"$PY" - <<'PYEOF'
import importlib
import os
import sys

# Adapters cd into the task dir, so several module-scope config loads are cwd-relative.
# These are the entrypoints the adapters actually invoke, plus the modules whose vLLM
# imports were made lazy.
CASES = [
    # tasks/prism_v1 is the only live bundle since 2026-09-04; the four originals are
    # untracked under tasks/archive/ and are not expected to import here.
    ("tasks/prism_v1", "run_inference"),
    ("tasks/prism_v1", "run_evaluation"),
    ("tasks/prism_v1", "src.driver"),
    ("tasks/prism_v1", "src.inference.aut_push"),
    ("tasks/prism_v1", "src.inference.creative_writing"),
    ("tasks/prism_v1", "src.inference.creative_short_story"),
    ("tasks/prism_v1", "src.inference.ttct"),
    ("tasks/prism_v1", "src.inference.creativity_index"),
    ("tasks/prism_v1", "src.inference.creative_math"),
    ("tasks/prism_v1", "src.inference.dat"),
    ("tasks/prism_v1", "src.inference.neocoder"),
    ("tasks/prism_v1", "src.evaluation.aut_push"),
    ("tasks/prism_v1", "src.evaluation.creative_writing"),
    ("tasks/prism_v1", "src.evaluation.creative_short_story"),
    ("tasks/prism_v1", "src.evaluation.ttct"),
    ("tasks/prism_v1", "src.evaluation.creativity_index"),
    ("tasks/prism_v1", "src.evaluation.creative_math"),
    ("tasks/prism_v1", "src.evaluation.dat"),
    ("tasks/prism_v1", "src.evaluation.neocoder"),
]

root = os.getcwd()
failed = 0
for rel, mod in CASES:
    os.chdir(os.path.join(root, rel))
    saved = list(sys.path)
    sys.path.insert(0, os.getcwd())
    # Task bundles all use the top-level names `src` and `configs`.
    for key in [
        k for k in sys.modules
        if k in ("src", "configs", "steps") or k.startswith(("src.", "configs.", "steps."))
    ]:
        del sys.modules[key]
    try:
        importlib.import_module(mod)
        print(f"  PASS: {rel}::{mod}")
    except Exception as exc:
        failed += 1
        print(f"  FAIL: {rel}::{mod} -> {type(exc).__name__}: {exc}")
    sys.path[:] = saved
    os.chdir(root)

print()
print(f"=== Summary: {len(CASES) - failed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
PYEOF
