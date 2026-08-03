#!/usr/bin/env bash
# Gate for result_analysis/loader.py.
#
# The mock judge used by the local smoke runs returns prose, so it can never
# produce a populated aut `cleaned_output`, a non-zero rubric verdict or a YES
# creative_math decision. Those paths are exercised here with synthetic fixtures
# whose shapes were frozen from real artifacts produced by the adapters, so this
# gate covers what an end-to-end run against the mock structurally cannot.
#
# Fixtures live in a temp directory; nothing here touches the repo's outputs/.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# A forced env or a redirected endpoint left over from a smoke run must not change
# how the loader reads files.
unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL

PASS=0
FAIL=0

check() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then
        printf '  ok   %s\n' "$label"
        PASS=$((PASS + 1))
    else
        printf '  FAIL %s\n         expected: %s\n         actual:   %s\n' \
            "$label" "$expected" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

# ---------------------------------------------------------------------------
# Build a synthetic outputs/ tree.
# ---------------------------------------------------------------------------
python - "$TMP_ROOT" <<'PYEOF'
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])

def run_dir(label, task, model):
    d = root / label / task / model
    d.mkdir(parents=True, exist_ok=True)
    return d

def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

def metadata(d, task, model, inference, evaluation):
    write(d / "metadata.json", {
        "label": d.parent.parent.name, "task": task, "inference_model": model,
        "exit_code": 0, "mode": "both",
        "artifacts": {"inference": {"native_path": str(inference)},
                      "eval": {"native_path": str(evaluation)}},
    })

# --- aut: the SCORED path the mock judge cannot produce -------------------
d = run_dir("fx", "aut", "M")
src = root / "_src" / "aut"
write(src / "inference_output.json", {"0": {"nc": "a bottle vase"},
                                      "1": {"nc": "a bottle lamp"}})
write(src / "eval_output_cleaned.json", [
    {"prompt_id": ["0", "nc"], "prompt_text": "JUDGE RUBRIC TEXT",
     "cleaned_output": [["vase", 4], ["lamp", 2]]},
    {"prompt_id": ["1", "nc"], "prompt_text": "JUDGE RUBRIC TEXT",
     "cleaned_output": []},
])
metadata(d, "aut", "M", src, src / "eval_output_cleaned.json")

# --- ttcw: non-zero rubric verdicts ---------------------------------------
d = run_dir("fx", "ttcw", "M")
src = root / "_src" / "ttcw"
write(src / "inference_output.json",
      {"s1": {"prompt_text": "write a story", "raw_output": "STORY"}})
write(src / "eval_output_cleaned.json", [
    {"prompt_id": ["s1", 1], "cleaned_output": 1},
    {"prompt_id": ["s1", 8], "cleaned_output": 0},
])
metadata(d, "ttcw", "M", src, src / "eval_output_cleaned.json")

# --- creative_math: a YES decision and a missing criterion ----------------
d = run_dir("fx", "creative_math", "M")
src = root / "_src" / "cmath"
write(src / "inference.json", [
    {"problem_id": "P1", "question_number": 0, "problem": "Q",
     "cleaned_response": "A"},
])
write(src / "eval.json", [
    {"problem_id": "P1", "question_number": 0,
     "correctness": {"final_decision": "YES"},
     "coarse_grained_novelty": {"final_decision": "NO"},
     "fine_grained_novelty": {}},
])
metadata(d, "creative_math", "M", src / "inference.json", src / "eval.json")

# --- neocoder: CSV evaluation, keyed join, evaluator drops a round --------
d = run_dir("fx", "neocoder", "M")
src = root / "_src" / "neo"
write(src / "inference" / "gen.json", [
    {"problem_id": "X1",
     "problem_statements": ["p0", "p1", "p2"],
     "outputs": ["o0", "o1", "o2"]},
])
csv_dir = src / "evaluation" / "creativity"
csv_dir.mkdir(parents=True, exist_ok=True)
# dp=1 is deliberately absent, and dp rows are out of order, so a positional
# join would produce different numbers than a keyed one.
(csv_dir / "c.csv").write_text(
    "problem_id,dp,correctness,follow_constraints,new_techniques_ratio\n"
    "X1,2,True,False,0.5\n"
    "X1,0,False,True,0\n",
    encoding="utf-8")
metadata(d, "neocoder", "M", src / "inference", csv_dir)
PYEOF

if [[ $? -ne 0 ]]; then
    echo "FATAL: could not build fixtures"
    exit 1
fi

# ---------------------------------------------------------------------------
# Assertions.
# ---------------------------------------------------------------------------
q() {
    python - "$TMP_ROOT" "$@" <<'PYEOF' 2>/dev/null
import sys, warnings
sys.path.insert(0, "result_analysis")
warnings.simplefilter("ignore")
import loader
root, task, expr = sys.argv[1], sys.argv[2], sys.argv[3]
rows = loader.load_records("fx", task=task, outputs_root=root)
print(eval(expr, {"rows": rows, "loader": loader, "len": len, "sum": sum,
                  "sorted": sorted, "set": set, "str": str, "repr": repr,
                  "None": None}))
PYEOF
}

echo "== aut: scored path =="
check "one row per scored use case, plus the unscored sample" \
    "3" "$(q aut 'len(rows)')"
check "scores come through as numbers" \
    "[2.0, 4.0]" "$(q aut 'sorted(r["eval_score"] for r in rows if r["eval_score"] is not None)')"
check "metric is novelty on scored rows" \
    "2" "$(q aut 'sum(1 for r in rows if r["metric"] == "novelty")')"
check "the judge rubric never leaks into the prompt column" \
    "0" "$(q aut 'sum(1 for r in rows if r["prompt"] is not None)')"
check "the empty-cleaned_output sample survives unscored" \
    "1" "$(q aut 'sum(1 for r in rows if r["metric"] is None)')"

echo "== ttcw: rubric verdicts =="
check "one row per rubric question" "2" "$(q ttcw 'len(rows)')"
check "question id becomes the metric" \
    "['rubric_q1', 'rubric_q8']" "$(q ttcw 'sorted(r["metric"] for r in rows)')"
check "a 1 verdict is not confused with a missing one" \
    "[0.0, 1.0]" "$(q ttcw 'sorted(r["eval_score"] for r in rows)')"
check "prompt comes from inference, not evaluation" \
    "2" "$(q ttcw 'sum(1 for r in rows if r["prompt"] == "write a story")')"

echo "== creative_math: three criteria =="
check "one row per criterion, always three" "3" "$(q creative_math 'len(rows)')"
check "YES becomes 1.0 and NO becomes 0.0" \
    "[0.0, 1.0]" "$(q creative_math 'sorted(r["eval_score"] for r in rows if r["eval_score"] is not None)')"
check "an empty criterion block stays an unscored row, not a dropped one" \
    "1" "$(q creative_math 'sum(1 for r in rows if r["metric"] == "fine_grained_novelty" and r["eval_score"] is None)')"

echo "== neocoder: CSV join =="
check "3 metrics for each of 2 scored rounds, plus 1 dropped round" \
    "7" "$(q neocoder 'len(rows)')"
check "the round the evaluator dropped stays unscored" \
    "1" "$(q neocoder 'sum(1 for r in rows if r["metric"] is None)')"
check "join is on dp, not on row order" \
    "1.0" "$(q neocoder '[r["eval_score"] for r in rows if r["sample_id"]=="X1|2" and r["metric"]=="correctness"][0]')"
check "a False in the CSV is 0.0, not None" \
    "0.0" "$(q neocoder '[r["eval_score"] for r in rows if r["sample_id"]=="X1|0" and r["metric"]=="correctness"][0]')"
check "text is joined from inference by the same key" \
    "3" "$(q neocoder 'sum(1 for r in rows if r["sample_id"]=="X1|2" and r["prompt"]=="p2")')"
check "every row carries prompt and output" \
    "0" "$(q neocoder 'sum(1 for r in rows if r["prompt"] is None or r["output"] is None)')"

echo "== schema =="
check "every row has exactly the declared columns" \
    "0" "$(q aut 'sum(1 for r in rows if set(r) != set(loader.COLUMNS))')"
check "every task in the registry has a parser" "0" \
    "$(python -c "
import sys, glob, os; sys.path.insert(0,'result_analysis')
import loader
names = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob('registry/tasks/*.yaml')}
print(len(names - set(loader.PARSERS)))")"

echo
echo "Summary: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
