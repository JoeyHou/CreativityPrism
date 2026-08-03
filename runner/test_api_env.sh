#!/usr/bin/env bash
# API-env gate: every task module must import without vLLM installed.
# Run from repo root: bash runner/test_api_env.sh
#
# Skips cleanly when the api env is absent, so it is safe to chain after the
# other gates on machines that only have the conda envs.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
    echo "SKIP: api env not found at $VENV (bash scripts/setup_envs.sh --env api)"
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
    ("tasks/aut_ttcw_cshort", "run_inference"),
    ("tasks/aut_ttcw_cshort", "run_evaluation"),
    ("tasks/aut_ttcw_cshort", "src.driver"),
    ("tasks/aut_ttcw_cshort", "src.evaluation.creative_writing"),
    ("tasks/math_n_index", "run_inference_all"),
    ("tasks/math_n_index", "run_inference"),
    ("tasks/math_n_index", "src.evaluation.creative_math_eval_api"),
    ("tasks/math_n_index", "src.inference.inference_driver"),
    ("tasks/math_n_index", "src.inference.creative_math_generation"),
    ("tasks/math_n_index", "src.inference.creative_index_generation"),
    ("tasks/math_n_index", "src.evaluation.creative_math_evaluation"),
    ("tasks/math_n_index", "src.models.local_models_vllm"),
    ("tasks/neocoder_dat", "steps.inference_dat"),
    ("tasks/neocoder_dat", "steps.evaluate_dat"),
    ("tasks/neocoder_dat", "steps.inference_dp"),
    ("tasks/neocoder_dat", "src.models.model"),
    ("tasks/neocoder_dat", "src.generator.dp_generator"),
    ("tasks/neocoder_dat", "src.generator.p_generator"),
    ("tasks/neocoder_dat", "src.utils.dat_score"),
    ("tasks/ttct", "src.inference.ttct_inference"),
    ("tasks/ttct", "src.evaluation.ttct_evaluation"),
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
