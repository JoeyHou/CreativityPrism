#!/usr/bin/env bash
# Adapter for the contributed Metaphor Generation task.
#
# Kept as a worked example of integrating a task that does NOT follow the
# built-in bundle's conventions: its own CLI, its own flag names, CSV output,
# and an output directory it names from a timestamp at run time. Nothing about
# the task was changed to make this work except its credential handling, which
# could not be worked around from outside (see README).
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

# NOT lookup_alias. That maps a canonical name to the per-task alias the
# built-in bundle's own model table expects; a task that goes through model_hub
# wants the canonical name itself. Mixing the two gives "Model 'gpt-4.1-mini'
# not in registry/models.yaml" while the canonical GPT4.1-mini sits right there.
ALIAS="$MODEL"
TASK_DIR="$REPO_ROOT/tasks/contrib_metaphor"
cd "$TASK_DIR"                 # the scripts read data/ relative to the cwd

LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--num-examples $LIMIT"

MARKER="$(mktemp)"
trap 'rm -f "$MARKER"' EXIT

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    touch "$MARKER"
    run_stage python scripts/gen.py --model_name "$ALIAS" $LIMIT_ARG
    if [[ -z "${CREATIVITYPRISM_ADAPTER_DRYRUN:-}" ]]; then
        # The task names its own directory, so bound the search by the marker
        # rather than trusting "most recent" -- two runs can overlap.
        GEN_CSV="$(newest_since "$MARKER" results/*/gen_*.csv)"
        [[ -n "$GEN_CSV" ]] || { echo "no generation csv appeared under results/" >&2; exit 1; }
        emit_artifact inference "$TASK_DIR/${GEN_CSV#./}"
    fi
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    JUDGE_ID="$JUDGE"   # canonical, for the same reason
    if [[ -z "${CREATIVITYPRISM_ADAPTER_DRYRUN:-}" ]]; then
        GEN_CSV="${GEN_CSV:-$(newest_since "$MARKER" results/*/gen_*.csv)}"
        [[ -n "$GEN_CSV" ]] || { echo "no generation csv to score" >&2; exit 1; }
    else
        GEN_CSV="results/<stamp>/gen_<stamp>.csv"
    fi
    touch "$MARKER"
    run_stage python scripts/score.py --infile "$GEN_CSV" --judge "$JUDGE_ID"
    if [[ -z "${CREATIVITYPRISM_ADAPTER_DRYRUN:-}" ]]; then
        emit_artifact eval "$TASK_DIR/$(dirname "$GEN_CSV")/originality_scores.json"
    fi
fi
