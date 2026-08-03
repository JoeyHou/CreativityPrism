#!/usr/bin/env bash
# Create environments listed under registry/environments/.
#   <name>.yml               -> conda env create
#   <name>.txt               -> conda create --file
#   <name>.requirements.txt  -> python -m venv + pip install -r  (no conda needed)
#
# Examples:
#   bash scripts/setup_envs.sh                # all envs in default location
#   bash scripts/setup_envs.sh --env modern   # one env only
#   bash scripts/setup_envs.sh --env api      # venv-backed, works without conda
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
            sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# conda is only required by the conda-backed specs; checked per env in create_env.

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

# venv envs always live on disk, so they are addressed by path even without --prefix.
venv_target() {
    local short="$1"
    if [[ -n "$PREFIX" ]]; then
        echo "${PREFIX%/}/creativityprism-${short}"
    else
        echo "$ENV_DIR/creativityprism-${short}"
    fi
}

create_venv_env() {
    local short="$1" spec="$2"
    local target; target="$(venv_target "$short")"

    if [[ -d "$target" && "$FORCE" -eq 0 ]]; then
        echo "  exists: $target (use --force to recreate)"
        return 0
    fi
    if [[ -d "$target" && "$FORCE" -eq 1 ]]; then
        echo "  removing existing $target"
        rm -rf "$target"
    fi

    local py=""
    for cand in python3 python py; do
        command -v "$cand" >/dev/null 2>&1 || continue
        "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null || continue
        py="$cand"; break
    done
    if [[ -z "$py" ]]; then
        echo "  no python >= 3.10 found in PATH" >&2
        return 3
    fi

    echo "  creating venv $target from $spec (python: $($py --version 2>&1))"
    "$py" -m venv "$target"

    # Windows venvs use Scripts/, POSIX venvs use bin/.
    local vpy="$target/bin/python"
    [[ -x "$vpy" ]] || vpy="$target/Scripts/python.exe"
    if [[ ! -x "$vpy" ]]; then
        echo "  venv created but no python found under $target" >&2
        return 3
    fi

    "$vpy" -m pip install --upgrade pip
    if ! "$vpy" -m pip install -r "$spec"; then
        echo "  pip install failed for $target" >&2
        case "$(uname -s)" in
            MINGW*|MSYS*|CYGWIN*)
                echo "  On Windows this is often the 260-character path limit." >&2
                echo "  Retry with a short location, e.g.: --prefix C:/cp-envs" >&2
                ;;
        esac
        return 3
    fi

    # Adapters invoke `python3`, which a Windows venv does not provide.
    if [[ "$vpy" == *"/Scripts/python.exe" && ! -e "$target/Scripts/python3.exe" ]]; then
        cp "$vpy" "$target/Scripts/python3.exe"
        echo "  added Scripts/python3.exe shim"
    fi

    # creative_short sentence-splits with nltk, which needs its corpus at run time.
    if "$vpy" -c "import nltk" >/dev/null 2>&1; then
        "$vpy" -m nltk.downloader punkt punkt_tab >/dev/null 2>&1 \
            || echo "  note: nltk corpus fetch failed; run '$vpy -m nltk.downloader punkt punkt_tab'" >&2
    fi

    # creative_short's story_metrics loads these by name, so a bare pip install is not enough.
    if "$vpy" -c "import spacy" >/dev/null 2>&1; then
        "$vpy" -m spacy download en_core_web_sm >/dev/null 2>&1 \
            || echo "  note: spacy model fetch failed; run '$vpy -m spacy download en_core_web_sm'" >&2
    fi
    if "$vpy" -c "import benepar" >/dev/null 2>&1; then
        "$vpy" -c "import benepar; benepar.download('benepar_en3')" >/dev/null 2>&1 \
            || echo "  note: benepar model fetch failed; run '$vpy -c \"import benepar; benepar.download(benepar_en3)\"'" >&2
    fi
}

create_env() {
    local short="$1"

    # venv spec takes priority: it is the only backend that works without conda.
    local req="$ENV_DIR/${short}.requirements.txt"
    if [[ -f "$req" ]]; then
        create_venv_env "$short" "$req"
        return $?
    fi

    local target; target="$(env_target "$short")"

    # Prefer .yml (conda env create) over .txt (conda create --file).
    local yml="$ENV_DIR/${short}.yml"
    local txt="$ENV_DIR/${short}.txt"
    if [[ -f "$yml" ]]; then
        local spec="$yml"; local use_yml=1
    elif [[ -f "$txt" ]]; then
        local spec="$txt"; local use_yml=0
    else
        echo "  skip: no $short.yml, $short.txt or $short.requirements.txt found" >&2
        return 0
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "  conda not found in PATH; '$short' needs it" >&2
        return 3
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

# Collect unique env names from every spec flavour.
shopt -s nullglob
declare -A seen
for f in "$ENV_DIR"/*.yml "$ENV_DIR"/*.txt; do
    short="$(basename "$f")"
    short="${short%.requirements.txt}"; short="${short%.yml}"; short="${short%.txt}"
    [[ "${seen[$short]:-}" ]] && continue
    seen[$short]=1
    if [[ -n "$ONLY_ENV" && "$short" != "$ONLY_ENV" ]]; then
        continue
    fi
    echo "==> $short"
    create_env "$short"
done

echo "Done."
