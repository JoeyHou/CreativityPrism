#!/usr/bin/env bash
# Phase 3 gate: SLURM script generation and artifact-marker durability.
#
# Everything here runs without a cluster and without a paid API call. Real sbatch
# submission is deliberately NOT covered: it cannot be exercised on a laptop.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Inherited overrides would silently change which env and endpoint are used.
unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL

PASS=0
FAIL=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS  $name"; PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"; FAIL=$((FAIL + 1))
    fi
}

check_contains() {
    local name="$1" file="$2" needle="$3"
    if grep -qF -- "$needle" "$file"; then
        echo "  PASS  $name"; PASS=$((PASS + 1))
    else
        echo "  FAIL  $name (missing: $needle)"; FAIL=$((FAIL + 1))
    fi
}

check_absent() {
    local name="$1" file="$2" needle="$3"
    if grep -qF -- "$needle" "$file"; then
        echo "  FAIL  $name (unexpected: $needle)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS  $name"; PASS=$((PASS + 1))
    fi
}

LABEL="phase3gate"
SCRIPT="slurm_scripts/${LABEL}/aut_GPT4.1.sbatch"
rm -rf "slurm_scripts/${LABEL}"

echo "== sbatch generation =="
check "--slurm --no-submit exits 0" \
    python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1 \
        --label "$LABEL" --limit 5 --slurm --no-submit
check "script written" test -f "$SCRIPT"

if [[ -f "$SCRIPT" ]]; then
    check_contains "shebang first line" "$SCRIPT" '#!/bin/bash'
    check_contains "job-name derived from label/task/model" "$SCRIPT" \
        '#SBATCH --job-name=cp-phase3gate-aut-GPT4.1'
    check_contains "log dir defaulted" "$SCRIPT" \
        '#SBATCH --output=slurm_scripts/phase3gate/logs/%x-%j.out'
    check_contains "template default carried through" "$SCRIPT" '#SBATCH --gres=gpu:1'
    check_contains "inner command re-invokes the runner" "$SCRIPT" 'runner/run.py'
    check_contains "--limit preserved into the job" "$SCRIPT" '--limit 5'
    check_absent "--slurm stripped from inner command" "$SCRIPT" '--slurm '
    check_absent "--no-submit stripped from inner command" "$SCRIPT" '--no-submit'
    # A script generated on one machine has to run on another, so no local path
    # may be baked in and the interpreter must be resolved from PATH.
    check_absent "no Windows drive path baked in" "$SCRIPT" 'C:\'
    check_absent "no absolute interpreter path" "$SCRIPT" 'python.exe'
    check_contains "repo root derived from script location" "$SCRIPT" 'readlink -f'
    check_absent "template-only comments stripped" "$SCRIPT" '#~'
    # The header is only honoured before the first executable line.
    FIRST_EXEC="$(grep -n -m1 -v -e '^#' -e '^[[:space:]]*$' "$SCRIPT" | cut -d: -f1)"
    LAST_SBATCH="$(grep -n '^#SBATCH ' "$SCRIPT" | tail -1 | cut -d: -f1)"
    check "all #SBATCH lines precede the first command" \
        test "$LAST_SBATCH" -lt "$FIRST_EXEC"
fi

echo "== directive overrides =="
rm -rf "slurm_scripts/${LABEL}"
check "--slurm-override accepted" \
    python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1 \
        --label "$LABEL" --slurm --no-submit \
        --slurm-override gres= --slurm-override time=1:30:00
if [[ -f "$SCRIPT" ]]; then
    check_contains "override applied" "$SCRIPT" '#SBATCH --time=1:30:00'
    check_absent "empty override drops the directive" "$SCRIPT" '--gres'
fi

check "malformed --slurm-override is rejected" \
    bash -c '! python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1 \
        --label '"$LABEL"' --slurm --no-submit --slurm-override nonsense'

echo "== one job per task =="
rm -rf "slurm_scripts/${LABEL}"
python runner/run.py --task all --model GPT4.1 --judge-model GPT4.1 \
    --label "$LABEL" --slurm --no-submit >/dev/null 2>&1
NJOBS="$(ls "slurm_scripts/${LABEL}"/*.sbatch 2>/dev/null | wc -l)"
NTASKS="$(ls registry/tasks/*.yaml | grep -vc '_task.schema' || true)"
check "--task all fans out to one script per task ($NJOBS)" test "$NJOBS" -eq "$NTASKS"
check_contains "fanned-out job targets a single task" \
    "slurm_scripts/${LABEL}/ttct_GPT4.1.sbatch" '--task ttct'
rm -rf "slurm_scripts/${LABEL}"

echo "== artifact marker durability =="
# emit_artifact must announce on stdout AND leave a sidecar, because under SLURM
# the runner never sees the job's stdout.
OUT_DIR="$TMP/out"
STDOUT="$(
    cd "$REPO_ROOT" && OUTPUT_DIR="$OUT_DIR" bash -c '
        source registry/adapters/_common.sh
        emit_artifact inference "'"$TMP"'/infer.json"
        emit_artifact eval "'"$TMP"'/eval.json"
    ' 2>/dev/null
)"
echo "$STDOUT" > "$TMP/stdout.txt"
check_contains "stdout marker still emitted" "$TMP/stdout.txt" 'CP_ARTIFACT inference'
check "sidecar written" test -f "$OUT_DIR/.cp_artifacts"
if [[ -f "$OUT_DIR/.cp_artifacts" ]]; then
    check_contains "sidecar records inference" "$OUT_DIR/.cp_artifacts" 'CP_ARTIFACT inference'
    check_contains "sidecar records eval" "$OUT_DIR/.cp_artifacts" 'CP_ARTIFACT eval'
fi

check "emit_artifact still works with no OUTPUT_DIR" \
    bash -c 'cd "'"$REPO_ROOT"'" && unset OUTPUT_DIR && source registry/adapters/_common.sh && emit_artifact inference /tmp/x.json'

check "sidecar parsed when stdout is empty" python - "$OUT_DIR" <<'PYEOF'
import sys
sys.path.insert(0, "runner")
import artifacts
found, warnings = artifacts.read_sidecar_markers(sys.argv[1])
assert set(found) == {"inference", "eval"}, found
assert not warnings, warnings
PYEOF

check "missing sidecar is not an error" python - <<'PYEOF'
import sys
sys.path.insert(0, "runner")
import artifacts
found, warnings = artifacts.read_sidecar_markers("does/not/exist")
assert found == {} and warnings == [], (found, warnings)
PYEOF

echo
echo "Phase 3: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
