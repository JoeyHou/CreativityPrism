#!/usr/bin/env bash
# Create conda environments listed under registry/environments/.
# Prefers <name>.yml (conda env create) over <name>.txt (conda create --file).
#
# Examples:
#   bash scripts/setup_envs.sh                # all envs in default conda location
#   bash scripts/setup_envs.sh --env modern   # one env only
#   bash scripts/setup_envs.sh --prefix /external/conda_envs
#   CREATIVITYPRISM_CONDA_PREFIX=/external/conda_envs bash scripts/setup_envs.sh
#   bash scripts/setup_envs.sh --env modern --force
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="$REPO_ROOT/registry/environments"
LOCATION_FILE="$ENV_DIR/.location"

ONLY_ENV=""
FORCE=0
PREFIX="${CREATIVITYPRISM_CONDA_PREFIX:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ONLY_ENV="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 3
fi

# Persist prefix for adapters/runner if provided.
if [[ -n "$PREFIX" ]]; then
    mkdir -p "$ENV_DIR"
    echo "$PREFIX" > "$LOCATION_FILE"
    echo "Wrote $LOCATION_FILE -> $PREFIX"
elif [[ -f "$LOCATION_FILE" ]]; then
    PREFIX="$(cat "$LOCATION_FILE")"
    echo "Using existing prefix from $LOCATION_FILE -> $PREFIX"
fi

env_target() {
    local short="$1"
    if [[ -n "$PREFIX" ]]; then
        echo "${PREFIX%/}/creativityprism-${short}"
    else
        echo "creativityprism-${short}"
    fi
}

env_exists() {
    local target="$1"
    if [[ "$target" = /* ]]; then
        [[ -d "$target" ]]
    else
        conda env list | awk '{print $1}' | grep -qx "$target"
    fi
}

create_env() {
    local short="$1"
    local target; target="$(env_target "$short")"

    # Prefer .yml (conda env create) over .txt (conda create --file).
    local yml="$ENV_DIR/${short}.yml"
    local txt="$ENV_DIR/${short}.txt"
    if [[ -f "$yml" ]]; then
        local spec="$yml"; local use_yml=1
    elif [[ -f "$txt" ]]; then
        local spec="$txt"; local use_yml=0
    else
        echo "  skip: no $short.yml or $short.txt found" >&2
        return 0
    fi

    if env_exists "$target" && [[ "$FORCE" -eq 0 ]]; then
        echo "  exists: $target (use --force to recreate)"
        return 0
    fi
    if env_exists "$target" && [[ "$FORCE" -eq 1 ]]; then
        echo "  removing existing $target"
        if [[ "$target" = /* ]]; then
            conda env remove -p "$target" -y
        else
            conda env remove -n "$target" -y
        fi
    fi

    echo "  creating $target from $spec"
    if [[ "$use_yml" -eq 1 ]]; then
        if [[ "$target" = /* ]]; then
            conda env create -f "$spec" -p "$target"
        else
            conda env create -f "$spec" -n "$target"
        fi
    else
        if [[ "$target" = /* ]]; then
            conda create -p "$target" --file "$spec" -y
        else
            conda create -n "$target" --file "$spec" -y
        fi
    fi
}

# Collect unique env names from both .yml and .txt files.
shopt -s nullglob
declare -A seen
for f in "$ENV_DIR"/*.yml "$ENV_DIR"/*.txt; do
    short="$(basename "$f")"
    short="${short%.yml}"; short="${short%.txt}"
    [[ "${seen[$short]:-}" ]] && continue
    seen[$short]=1
    if [[ -n "$ONLY_ENV" && "$short" != "$ONLY_ENV" ]]; then
        continue
    fi
    echo "==> $short"
    create_env "$short"
done

echo "Done."
