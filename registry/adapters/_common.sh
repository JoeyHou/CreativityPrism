#!/usr/bin/env bash
# Shared helpers for adapters. Source this from each adapter:
#   source "$(dirname "$0")/_common.sh"

set -euo pipefail

# Locate repo root from this file's location.
ADAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ADAPTER_DIR/../.." && pwd)"

# ---- Arg parsing ----
# Sets globals: MODEL, JUDGE, RUN_ID, OUTPUT_DIR, LIMIT, MODE
parse_adapter_args() {
    MODEL=""
    JUDGE=""
    RUN_ID=""
    OUTPUT_DIR=""
    LIMIT=""
    MODE="both"  # both | inference | eval
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model) MODEL="$2"; shift 2 ;;
            --judge-model) JUDGE="$2"; shift 2 ;;
            --run-id) RUN_ID="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --limit) LIMIT="$2"; shift 2 ;;
            --inference-only) MODE="inference"; shift ;;
            --eval-only) MODE="eval"; shift ;;
            *) echo "Unknown adapter arg: $1" >&2; exit 2 ;;
        esac
    done
    if [[ -z "$MODEL" || -z "$RUN_ID" || -z "$JUDGE" ]]; then
        echo "Adapter requires --model, --judge-model, and --run-id" >&2
        exit 2
    fi
}

# ---- Artifact contract ----
# Announce a produced artifact to the runner, which links it into
# outputs/<label>/<task>/<model>/ and records it in metadata.json.
#   emit_artifact inference "$NATIVE_OUT"
#   emit_artifact eval      "$EVAL_OUT"
# Emit a marker only for a phase that actually ran. Paths must be absolute.
emit_artifact() {
    local kind="$1"
    local path="$2"
    case "$kind" in
        inference|eval) ;;
        *) echo "emit_artifact: unknown kind '$kind'" >&2; return 2 ;;
    esac
    # The runner is Python, so hand it a path its os.path understands.
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) path="$(cygpath -w "$path")" ;;
    esac
    echo "CP_ARTIFACT ${kind} ${path}"
    # Under SLURM the adapter's stdout goes to a batch log the runner never reads, so the
    # same marker is also appended beside the run's outputs, where a later collect step
    # can find it. Harmless locally: the runner merges both sources and prefers stdout.
    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        mkdir -p "$OUTPUT_DIR" 2>/dev/null \
            && printf 'CP_ARTIFACT %s %s\n' "$kind" "$path" >> "${OUTPUT_DIR}/.cp_artifacts"
    fi
}

# ---- models.yaml lookup (pure python, no PyYAML required at adapter level) ----
# Usage: lookup_alias <canonical_model> <task_key>
# Echoes the alias to stdout, or exits non-zero if not found.
lookup_alias() {
    local canonical="$1"
    local task_key="$2"
    python3 - "$canonical" "$task_key" "$REPO_ROOT/registry/models.yaml" <<'PYEOF'
import sys, yaml
canonical, task_key, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    data = yaml.safe_load(f)
models = data.get("models", {})
if canonical not in models:
    sys.stderr.write(f"Model '{canonical}' not in registry/models.yaml\n")
    sys.exit(1)
aliases = models[canonical].get("aliases", {})
if task_key not in aliases:
    sys.stderr.write(f"Model '{canonical}' has no alias for task '{task_key}'\n")
    sys.exit(1)
print(aliases[task_key])
PYEOF
}

# ---- Cluster-specific environment setup ----
# Source registry/environments/.cluster_env.sh if present (git-ignored, site-local).
# Use this file for cluster-specific setup that must run before conda env activation:
#   - module load commands (Lmod/Environment Modules)
#   - cluster-specific LD_LIBRARY_PATH entries
#   - scheduler-specific exports
# See registry/environments/cluster_env.sh.example for a template.
_CLUSTER_ENV="$REPO_ROOT/registry/environments/.cluster_env.sh"
if [[ -f "$_CLUSTER_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$_CLUSTER_ENV"
fi
unset _CLUSTER_ENV

# ---- Credentials ----
# CREATIVITYPRISM_API_KEYS is set by the runner to the abs path of api_keys.json.
# Adapters export it so child processes (task scripts) can read it.
# Fall back to repo-root api_keys.json if the runner didn't set it.
if [[ -z "${CREATIVITYPRISM_API_KEYS:-}" ]]; then
    CREATIVITYPRISM_API_KEYS="$REPO_ROOT/api_keys.json"
fi
export CREATIVITYPRISM_API_KEYS

# tasks/neocoder_dat reads OPENAI_API_KEY / ANTHROPIC_API_KEY / GENAI_API_KEY /
# DEEPSEEK_API_KEY straight from the environment, and tasks/math_n_index was
# patched to do the same. Resolve them from the credentials file so no key is
# ever written into the repo tree. Keys already set in the environment win.
export_provider_keys() {
    [[ -f "${CREATIVITYPRISM_API_KEYS:-}" ]] || return 0
    local exports
    exports="$(python3 "$ADAPTER_DIR/_provider_keys.py" "$CREATIVITYPRISM_API_KEYS")" || return 0
    [[ -n "$exports" ]] && eval "$exports"
    return 0
}

# ---- PYTHONPATH ----
# Task bundles are imported as the top-level package `src`, so the task dir must be
# importable. Git Bash `pwd` yields "/c/...", which Windows Python cannot read, and
# there the separator is ';' rather than ':'.
add_pythonpath() {
    local dir="$1"
    local sep=":"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            dir="$(cygpath -w "$dir")"
            sep=";"
            ;;
    esac
    if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="${PYTHONPATH}${sep}${dir}"
    else
        export PYTHONPATH="$dir"
    fi
}

# ---- Conda env activation ----
# Reads registry/environments/.location if present.
activate_env() {
    local env_short="$1"  # e.g. modern

    # CREATIVITYPRISM_FORCE_ENV overrides the task's declared environment. Intended for
    # the API-only venv, where the GPU stack is absent but every API path still works.
    if [[ -n "${CREATIVITYPRISM_FORCE_ENV:-}" ]]; then
        env_short="$CREATIVITYPRISM_FORCE_ENV"
    fi
    # Adapters that need a GPU-free fallback branch on this.
    export CREATIVITYPRISM_ACTIVE_ENV="$env_short"

    local env_name="creativityprism-${env_short}"
    local prefix=""
    if [[ -n "${CREATIVITYPRISM_ENV_PREFIX:-}" ]]; then
        prefix="$CREATIVITYPRISM_ENV_PREFIX"
    elif [[ -f "$REPO_ROOT/registry/environments/.location" ]]; then
        prefix="$(cat "$REPO_ROOT/registry/environments/.location")"
    fi

    # A venv-backed env is a plain directory; conda is not involved.
    local venv_dir=""
    if [[ -n "$prefix" ]]; then
        venv_dir="${prefix%/}/${env_name}"
    else
        venv_dir="$REPO_ROOT/registry/environments/${env_name}"
    fi
    if [[ -x "${venv_dir}/bin/python" || -x "${venv_dir}/Scripts/python.exe" ]]; then
        local bin_dir="${venv_dir}/bin"
        [[ -x "${venv_dir}/Scripts/python.exe" ]] && bin_dir="${venv_dir}/Scripts"
        # PATH is colon-separated, so a "C:/..." entry would be torn in two under Git Bash.
        if command -v cygpath >/dev/null 2>&1; then
            bin_dir="$(cygpath -u "$bin_dir")"
        fi
        export PATH="${bin_dir}:${PATH}"
        export VIRTUAL_ENV="$venv_dir"
        unset PYTHONHOME
        return 0
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found in PATH" >&2
        exit 3
    fi

    # Resolve the env directory path.
    local env_dir=""
    if [[ -n "$prefix" ]]; then
        env_dir="${prefix%/}/${env_name}"
    else
        # Named env: ask conda where it lives.
        env_dir="$(conda info --base)/envs/${env_name}"
    fi

    if [[ ! -d "$env_dir" ]]; then
        echo "Environment not found: $env_dir" >&2
        echo "Run: bash scripts/setup_envs.sh --env ${env_short}" >&2
        exit 4
    fi

    # Prepend env bin/ to PATH — works in non-interactive shells unlike conda activate.
    export PATH="${env_dir}/bin:${PATH}"
    export CONDA_PREFIX="$env_dir"
    export CONDA_DEFAULT_ENV="$env_name"

    # Redirect Triton/torch compiler caches away from /ihome (quota-limited)
    # to the scratch space alongside the conda envs.
    local cache_base
    if [[ -n "$prefix" ]]; then
        cache_base="${prefix%/}/.cache"
    else
        cache_base="$env_dir/.cache"
    fi
    export TRITON_CACHE_DIR="${cache_base}/triton"
    export TORCH_HOME="${cache_base}/torch"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME"

    # ---- HuggingFace cache (HF_HOME) ----
    # CRITICAL: HF model downloads are large (tens of GB). Never let them land
    # in the home directory, which is typically quota-limited on HPC clusters.
    #
    # Priority (highest to lowest):
    #   1. HF_HOME already set in environment (user exported it before running)
    #   2. CREATIVITYPRISM_HF_HOME env var
    #   3. registry/environments/.hf_home file  (site-local, git-ignored)
    #   4. <conda-prefix-cache>/huggingface      (safe fallback on same scratch disk)
    #
    # To configure for your cluster, either:
    #   export CREATIVITYPRISM_HF_HOME=/path/to/scratch/huggingface
    # or write the path to registry/environments/.hf_home (git-ignored).
    if [[ -z "${HF_HOME:-}" ]]; then
        if [[ -n "${CREATIVITYPRISM_HF_HOME:-}" ]]; then
            export HF_HOME="$CREATIVITYPRISM_HF_HOME"
        elif [[ -f "$REPO_ROOT/registry/environments/.hf_home" ]]; then
            export HF_HOME="$(cat "$REPO_ROOT/registry/environments/.hf_home")"
        else
            export HF_HOME="${cache_base}/huggingface"
            echo "WARNING: HF_HOME not configured — defaulting to ${HF_HOME}" >&2
            echo "         Set CREATIVITYPRISM_HF_HOME or write registry/environments/.hf_home" >&2
            echo "         to redirect model downloads away from your home directory." >&2
        fi
    fi
    mkdir -p "$HF_HOME"

    # LD_LIBRARY_PATH: add pip-installed nvidia/nvjitlink if present in the env.
    # Needed on clusters where the system CUDA module doesn't include nvjitlink
    # (e.g. CRC with cuda/12.1). The nvidia-nvjitlink-cu12 pip package ships the
    # library; we discover its path from $env_dir so it's not hardcoded per-user.
    # Globbed because envs differ in Python version (modern 3.12, legacy 3.11).
    local nvjitlink_dir
    for nvjitlink_dir in "${env_dir}"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
        if [[ -d "$nvjitlink_dir" ]]; then
            export LD_LIBRARY_PATH="${nvjitlink_dir}:${LD_LIBRARY_PATH:-}"
            break
        fi
    done

    # vLLM multiprocessing: spawn avoids fork-related CUDA issues on some clusters.
    # Safe to set on all clusters; override with VLLM_WORKER_MULTIPROC_METHOD if needed.
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

    # Suppress vLLM telemetry writes (also go to /ihome).
    export VLLM_NO_USAGE_STATS=1

    # Prevent transformers from trying to import torchvision (not needed here,
    # and the import can fail or trigger warnings in some environments).
    export TRANSFORMERS_NO_TORCHVISION=1
}