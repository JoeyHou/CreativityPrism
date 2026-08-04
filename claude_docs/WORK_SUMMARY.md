# CreativityPrism Restructuring — Work Summary

**As of 2026-08-03.** This document stands alone. It does not assume you have read
`RESTRUCTURING_PLAN.md` (the design contract) or `CHANGE_LOG.md` (the chronological record),
and it deliberately does not repeat their structure: the plan is organised by *phase*, the
change log by *date*, and this by *what is now true*.

Read this if you want to know what the system does, what was wrong with it, what was decided
and why, and — the part that matters most for anyone about to trust a number that comes out of
it — **what the fixed code assumes**.

Operational instructions live in [RUNNING_GUIDE.md](RUNNING_GUIDE.md). Nothing here tells you
how to run anything.

---

## 1. What the system is now

CreativityPrism is eight creativity benchmarks that were written independently, by different
people, with different data formats, different CLIs, different model-naming conventions and two
mutually incompatible vLLM pins. The restructuring did **not** rewrite them. It put a single
seam in front of them and left the science untouched.

```
              registry/models.yaml        registry/tasks/{task}.yaml
                      │                            │
                      │  canonical name → alias    │  which env, which adapter
                      ▼                            ▼
  user ──►  runner/run.py  ──►  registry/adapters/{task}.sh  ──►  tasks/{bundle}/
                      ▲                            │
                      │      CP_ARTIFACT markers   │
                      └────────────────────────────┘
                      │
                      ▼
              outputs/{label}/{task}/{model}/metadata.json
                      │
                      ▼
              result_analysis/loader.py  ──►  one long table
```

Five properties are the whole point of the design:

1. **One command, one vocabulary.** `--task`, `--model`, `--judge-model`, `--label`. Model
   identity is canonical (`GPT4.1`); every bundle's private spelling (`gpt_4.1`,
   `gpt-4.1-2025-04-14`, `gpt-4.1`) is a lookup in `registry/models.yaml`, and a *missing*
   alias is an explicit failure rather than a guess.
2. **Tasks keep writing where they always wrote.** The runner never asks a bundle to change its
   output layout. It only asks the adapter to *announce* what it produced, with a
   `CP_ARTIFACT <kind> <path>` line on stdout.
3. **The runner never reads an artifact.** It links it into `outputs/` and records the native
   path in `metadata.json`. Parsing formats is the loader's job, so a task can change its file
   format without touching the runner.
4. **A cluster run and a laptop run produce identical `outputs/` trees.** The generated sbatch
   script re-invokes `runner/run.py` *inside* the job, not the adapter, so artifact linking
   happens on the compute node and nothing has to be collected afterwards.
5. **Analysis reads `outputs/` only.** `loader.py` resolves artifacts through `metadata.json`,
   so it knows every task's *file format* but no task's *directory layout*.

### The eight tasks

| Task | Domain | Bundle | Env | `--limit`? | Judge model used? |
|---|---|---|---|---|---|
| `aut` | divergent thinking | `tasks/aut_ttcw_cshort` | modern | yes | **yes** |
| `ttcw` | creative writing | `tasks/aut_ttcw_cshort` | modern | yes | **yes** |
| `creative_short` | creative writing | `tasks/aut_ttcw_cshort` | modern | yes | no — automatic metrics |
| `ttct` | divergent thinking | `tasks/ttct` | modern | yes (special) | **yes** |
| `neocoder` | logical reasoning | `tasks/neocoder_dat` | legacy | **no** | no — hardcodes `gpt-4-turbo` |
| `dat` | divergent thinking | `tasks/neocoder_dat` | legacy | yes (= `--repeat`) | no — GloVe distance |
| `creative_math` | logical reasoning | `tasks/math_n_index` | modern | yes | no — fixed 3-judge panel |
| `creativity_index` | divergent thinking | `tasks/math_n_index` | modern | yes | no — n-gram overlap |

`--judge-model` is required by the CLI for all eight even though five ignore it. That was a
deliberate choice: making it conditional would mean the *presence* of a flag silently changes
meaning per task, which is worse than an ignored argument.

---

## 2. Bugs found

They are grouped by consequence, because that is the only grouping that matters when deciding
what to re-run.

### 2.1 Bugs that change published numbers

| # | Where | What was wrong | Status |
|---|---|---|---|
| 1 | `extract_yes_no()`, 3 copies | Substring match | **Fixed** |
| 2 | `neocoder` correctness stage | Model code never actually ran | **Fixed** |
| 3 | `neocoder` comparison stage | Three independent faults | **Not fixed — needs a decision** |
| 4 | `creativity_index` infini-gram client | Lost lookups scored as "not in corpus" | **Fixed** |
| 5 | `creative_math` judge panel | An API failure is recorded as a `NO` vote | **Not fixed — documented** |
| 6 | `loader.py` `creative_math` join | Join key was not unique | **Fixed** |
| 7 | `loader.py` `ttct` | The judge's scores were never parsed | **Fixed** |
| 8 | Gemini wrappers | Empty response on 2.5+ models | **Fixed in 1 of 4** |

**1 — `extract_yes_no()` matched substrings.** It upper-cased the judge's reply and asked
`if "YES" in text ... elif "NO" in text`. `"NO"` is a substring of `NOT`, `CANNOT`, `KNOW`,
`NOTE` and `NOVEL`, so a judge that opened with *"I need to know whether…"* was recorded as a
`NO` vote — observed live. And because `"YES"` was tested first and anywhere in the string,
*"The answer is not YES"* scored `YES`. All three copies now take the first **whole-word**
`YES`/`NO`, case-insensitively; every prompt asks for the verdict first and the explanation
after, which is what makes "first token wins" correct. The vLLM copies additionally recognise
lower-case `yes`, which previously scored `NO`.

**2 — `neocoder` scored 0 of 60 because its harness could not run the code it was given.**
The harness patched `builtins.input` and executed the solution in-process. But 55 of the 60
generated solutions open with the standard competitive-programming idiom:

```python
def solve():
    import sys
    input = sys.stdin.read      # binds a LOCAL name — builtins is never consulted
    data = input().split()
```

`parse_code`'s six rewrite rules all require parentheses (`sys.stdin.read()`,
`.readline()`, `.readlines()`), so the paren-free assignment slipped through untouched. With an
inherited terminal stdin those 55 blocked forever — that is the entire "code execution timeout"
epidemic — and with `/dev/null` they read `''`, produced an empty list, and died on the first
subscript. `parse_code` and `exec` were innocent: all 60 solutions parsed, executed and defined
`solve()`.

**3 — `neocoder`'s comparison layer is broken in three separate ways**, discovered only once
the code could actually run. In the order they bite:

- The **first try can never compare anything.** It flattens every case's output into one list of
  lines and zips that against `test_case_outputs`, which is a list *per case*. So
  `type_agnostic_compare('9', ['9','10 1','15 5'])` evaluates `'9'` and then calls
  `list(map(str, 9))` → `TypeError`. The first try has never once produced a verdict.
- The **per-case retry feeds a malformed file.** A stored case looks like
  `[['2'], ['15','1','10','5']]` — that is `n` followed by the data, with **no leading test
  count**. The first try prepends one; the retry does not. A solution written for the
  multi-test format reads `n` as `t` and crashes.
- Even with both repaired, it is an **exact match against a single reference answer**, and these
  problems accept many. Problem `1895B` was inspected in full: the model emitted the same
  optimal cost with a different (equally optimal) pairing and was scored wrong. Codeforces
  judges this class of problem with a per-problem checker; NeoCoder ships none.

  > After repairing the first two mechanically, the result was 3 pass / 4 crash / 53 "differs
  > from reference". **How many of those 53 are genuine failures has not been measured.** Do not
  > assume either answer.

**4 — `creativity_index` degraded silently, in a flattering direction.**
`api.infini-gram.io` is a free public endpoint that answered `403 Forbidden` for roughly half of
all requests even at one per second. The retry loop used a flat 0.2 s delay over 5 attempts, so
every attempt landed inside the same throttle window; on exhaustion it fell through to
`occurrence = 0`, which is **the same value as "this n-gram does not appear in the corpus"**.
Lost lookups therefore lowered coverage, and lower coverage *raises* the reported creativity
index. It now uses 8 attempts with exponential backoff and jitter, and raises on exhaustion.
**Any `creativity_index` number computed before this fix is suspect.**

**5 — In `creative_math`, a dead judge does not look like an error.** Three of the four API
wrappers return the exception *as a string* instead of raising; that string is then compared
against `"YES"`, so any API failure is recorded as a `NO` vote. A real artifact in this repo
shows a Gemini `API_KEY_INVALID` message stored as `"gemini-2.0-flash": "NO"` with the error
text sitting in `reasons`. Correctness requires unanimity, so **one dead provider drives
correctness to 0%**, indistinguishable from wrong mathematics — and then cascades, because
correctness `NO` forces coarse novelty `NO`, which forces fine novelty `NO`. One retired model
ID can zero an entire run. Left as-is because fixing it means changing what the panel does when
a judge is unavailable, which is a benchmark-design decision.

**6 — the loader's `creative_math` join key collided.** The same problem is asked several times
with different `k` (the number of reference solutions shown), so `(problem_id, question_number)`
is not unique: an 18-record run collapsed to 10 distinct keys and every collision silently
inherited the last record's verdicts. That halved the reported correctness of the first real
run. The key now includes `k`.

**7 — `ttct` outputs were loaded but never scored.** The judge returns free text; the numbers
live under a `### Scores ###` heading. `extract_scores` from
`human_annotation/notebooks/mturk_agreement.ipynb` was ported verbatim into the loader rather
than rewritten, specifically so the pipeline's numbers and the human-agreement analysis's
numbers come from the same parser. Each judged item now yields four rows — Fluency,
Flexibility, Originality, Elaboration.

**8 — Gemini 2.5+ spends the output budget on thinking.** With a small `max_output_tokens` the
response returns `finishReason=MAX_TOKENS` with **no `parts` key at all**, so the wrapper raises
`KeyError` or returns empty text. Fixed with `thinking_budget=0` in
`tasks/math_n_index/api_warpper.py`, the judge path actually exercised. **Three wrappers are
still unfixed** — `aut_ttcw_cshort`, `ttct`, and `neocoder_dat`'s `GenAIModel`. Point a Gemini
model at those and it will silently return nothing.

### 2.2 Bugs that break runs without corrupting numbers

These are less interesting individually but between them they are the difference between "the
pipeline works" and "the pipeline appears to hang".

- **The adapter inherited an interactive stdin.** No task reads stdin, but several *execute
  model-generated code* that does. Now `stdin=subprocess.DEVNULL`, which turns an inherited
  prompt into an instant `EOFError` — which is what a batch scheduler would have produced anyway.
- **A finished run could be killed by its own output.** On Windows, Python defaults stdout to
  cp1252; a judge writing `✓` raised `UnicodeEncodeError` mid-print and aborted an evaluation
  that had already succeeded. `PYTHONIOENCODING=utf-8` is now forced for the child and the
  parent's stdout is reconfigured to match.
- **A one-hour task looked completely hung**, because its stdout is a pipe and Python
  block-buffers pipes. `PYTHONUNBUFFERED=1`.
- **The correctness timeout could not fire**, because the worker thread was non-daemon and the
  interpreter waited for it at exit. `thread.daemon = True`.
- **Windows has no symlinks without Developer Mode.** The runner writes a one-line
  `{kind}_output.path` reference instead and records `link_type: reference`;
  `metadata.json.artifacts.<kind>.native_path` is populated identically on both platforms, which
  is why analysis code must read that and never assume a symlink.
- **Windows venvs ship `Scripts/python.exe` but no `python3.exe`**, which every adapter calls; a
  copy is made at setup. And a `C:/...` entry prepended to a colon-separated `PATH` would be
  torn in two, so `activate_env` converts to POSIX form first.

---

## 3. Decisions made, and what was rejected

Each of these was load-bearing enough that reversing it later would be expensive.

**The adapter is a shell script, not a Python plugin.** Every bundle already had a working shell
invocation. A plugin interface would have meant importing eight mutually incompatible dependency
trees into one process — including two vLLM pins that cannot coexist. *Rejected:* a Python entry
point per task.

**Artifacts are announced on stdout, not discovered by convention.** A convention
(`look in tasks/*/output/{label}/`) would force every bundle to adopt a layout, which is exactly
the rewrite this restructuring exists to avoid. The marker also survives SLURM: `emit_artifact`
additionally appends to `{output_dir}/.cp_artifacts`, because under `sbatch` the adapter's stdout
goes to a batch log the runner never reads. *Rejected:* path conventions, and a post-hoc collect
step.

**The sbatch script wraps `runner/run.py`, not the adapter.** So the job does its own artifact
linking and metadata writing. *Rejected:* wrapping the adapter and collecting afterwards, which
would have made cluster and laptop runs produce different trees.

**`eval_score` is meaningless without `metric`, so `metric` is a column.** A DAT semantic
distance around 86, an n-gram coverage in [0, 1] and a binary judge verdict are not the same
quantity. *Rejected:* one `score` column, which would have silently averaged them.

**The loader flattens and never aggregates.** One row per scored unit. `aut` scores every
extracted use, `creativity_index` every n-gram size, `neocoder` every denial round,
`creative_math` returns three verdicts per problem. What counts as "the score" of a task is an
analysis decision that belongs in the notebook. *Rejected:* returning task-level means.

**A generated-but-unscored unit is kept as a row with `metric = None`.** So the row count
reflects what *ran*, not what the evaluator managed to score. Dropping them would have hidden
exactly the neocoder failure described above. *Rejected:* filtering to scored rows.

**pandas is optional.** `load_records()` returns dicts and imports nothing; `load_outputs()`
imports pandas lazily. This is why the loader gate can run in an interpreter without pandas.

**`ttct` defaults to the `cot` prompt variant only — the other two are kept, not deleted.**
Only `cot` is ever judged, so generating `basic` and `instructive` cost money and produced
nothing scoreable: a full run was 1500 API calls where 500 were used. The obvious fix is to
delete the two variants; that was **rejected** after checking what depends on them:

- `csv2json` and `json2csv` in the ttct bundle address six `infer_*` columns *by name*.
- 45 committed files under `human_annotation/data/mturk_anno/ttct/v1.1_llmj_output/` contain
  **real** basic and instructive generations — 675 of each, measured, not assumed — and they
  back the human-agreement analysis.

So the columns stay and unrequested variants are written as the literal string `SKIPPED`. The
schema is fixed, the corpora stay valid, and the cost drops 3×.

> A related trap: `assert len(csv_data) == len(input_data)` in `ttct_evaluation.py` compares
> **row** counts (700). The variants are **columns**. It is not a guard against variant removal,
> though it looks like one.

**A default `ttct` run is 500 items, not 700.** The dataset ships 7 question types × 100, but
the LLM-judge rubric was aligned against human ratings for five of them only
(`1_unusual_uses`, `2_consequences`, `4_situation`, `5_common_problems`, `6_improvement`).
`3_just_suppose` and `7_story` ship for completeness and stay unscored; their 200 rows remain in
the file as `SKIPPED`.

**`neocoder` solutions run in a subprocess (option b), not behind a smarter `input` patch.**
Extending the rewrite to catch `input = sys.stdin.read` (option a) needs a closure, because
`sys.stdin.read` returns everything at once while the mock yields lines — and it would still be
a blocklist of idioms, defeated by the next one. A real pipe is what Codeforces gives the
solution, it makes `input()`, `sys.stdin.read` and `sys.stdin.readline` all behave alike without
enumerating them, and a subprocess can actually be killed on timeout where a thread cannot. It
also stops the evaluator `exec`-ing solutions into its own globals, so one problem's `solve()`
can no longer leak into the next. *Rejected:* option (a).

**The `creative_math` cleaner is a fixed instrument, independent of the model under test.**
Letting it track the model being evaluated would make API-model and open-model scores
non-comparable. It defaults to vLLM + Llama-3.3-70B on 4 GPUs, the published setup; the API env
cannot host that, so `creative_math --eval-only` there **exits 4 with instructions** rather than
silently substituting a smaller cleaner. Each item records `cleaner_model`, so a mixed set is
detectable after the fact.

**`--limit` is opt-in per task, and rejected loudly where unsupported.** `neocoder` exposes no
sample-count knob, so `--limit` there is an error rather than a silent no-op, and
`--task all --limit N` is rejected outright. For `dat`, `--limit` maps to `--repeat`, because DAT
has one fixed prompt rather than a dataset — the repeat count *is* the sample count.

**`ttct --limit N` marks rows `skip` instead of truncating**, because the eval phase asserts one
row per `basefile.csv` row. It keeps all 700 and queries only the first N of each scored type, so
`--limit 2` issues 10 calls and scores 10 items.

---

## 4. What the changed code assumes

This is the section to read before trusting a number. Each assumption is stated with what
happens when it is false.

### 4.1 `run_solution()` — the neocoder execution fix

| Assumption | If false |
|---|---|
| The solution is a self-contained script that reads stdin and writes stdout. | A solution that only defines a helper for an external driver produces empty stdout and is scored wrong, not errored. |
| Appending `solve()` is the right driver when `parse_code` kept no top-level call. | A model that named its entry point something else runs nothing → empty stdout → scored wrong. Detection is a column-0 regex, which is sound *because* `parse_code` truncates at the first line that is neither indented nor starts with `solve` — so an `if __name__ == "__main__":` guard and its call are already stripped before the check, and appending a driver is the only way to get one back. |
| `sys.executable` is an appropriate interpreter for model code. | It is the evaluator's own interpreter. A solution importing `numpy` succeeds or fails depending on the env, so **scores are not reproducible across environments** with different packages installed. |
| 6 seconds per attempt is enough. | A correct-but-slow solution is scored as failed. The value is a constant, `SOLUTION_TIMEOUT`. |
| A non-zero exit code means failure. | A solution that prints the right answer and then raises during cleanup is scored failed. |
| stdout/stderr are UTF-8. | Forced via `PYTHONIOENCODING`; decoding errors are replaced, not raised. |
| **Executing untrusted model-generated code is acceptable here.** | A subprocess is **not a sandbox** — no seccomp, no network isolation, no filesystem jail beyond `cwd=tmpdir`. The trust level is unchanged from before, but the code now genuinely executes, so the exposure is real for the first time. Run `neocoder` only where that is acceptable. |

### 4.2 `ttct` defaults

- Assumes nothing downstream needs `basic`/`instructive` **at run time**. The columns still
  exist, filled with `SKIPPED`; the committed human-annotation corpora are untouched.
- Assumes the judge is still invoked for `SKIPPED` rows and returns quickly. **This was left
  alone deliberately** — it is a wasted call, not a wrong number.
- Assumes the five-type subset is the intended default. Override with `-subset`.

### 4.3 `_parse_ttct_scores()` — the loader's ttct parser

- Assumes the judge emits a `### Scores ###` heading followed by `Trait: N` lines. This is
  ported verbatim rather than generalised, so its behaviour matches the human-agreement notebook
  exactly.
- Assumes only the four known traits count; anything else the regex catches is dropped.
- **A judge that changes its output format yields `metric = None` rows, not wrong numbers.** The
  failure is visible in the row count rather than hidden in a mean.

### 4.4 `extract_yes_no()`

- Assumes the verdict appears **before** the explanation. Verified by reading the prompts, all
  of which ask for it first. A prompt rewritten to ask for reasoning first would silently
  invert results.
- Each copy keeps its original fallback, so `api_eval.py` still returns `UNCLEAR` and the vLLM
  copies still return `NO`. Every downstream tally already counts `UNCLEAR` exactly like `NO`.

### 4.5 The artifact contract

- Assumes the adapter emits a marker **only** for a phase that actually ran and succeeded. An
  `--eval-only` invocation must not announce an inference artifact. The runner does not verify
  this; it cannot, because it never reads the contents.
- Assumes announced paths are absolute. Under Git Bash they are converted with `cygpath -w`,
  because the consumer is Python.
- Assumes stdout markers beat sidecar markers, since a stale `.cp_artifacts` can survive from an
  earlier run into the same `(label, task, model)` directory.

### 4.6 The loader

- `_reroot()` assumes **every task writes somewhere under `tasks/`**. That is the anchor that
  lets a run performed on the cluster be analysed on a laptop. A task writing to `/scratch/...`
  outside the repo would break cross-machine analysis.
- Assumes `metadata.json` is authoritative and a symlink may not exist.
- Assumes a task with no registered parser is a *warning*, not a failure — unless
  `strict=True`.

### 4.7 Environments

- `CREATIVITYPRISM_FORCE_ENV=api` assumes no open-weight model is involved; that combination is
  rejected up front rather than failing deep inside task code.
- `activate_env` prepends the env's `bin/` to `PATH` instead of calling `conda activate`, which
  does not work in a non-interactive shell.

---

## 5. Verification status — what is proven and what is not

Everything was built and verified on **one Windows laptop against hosted APIs**. That machine
has no conda, no GPU, no symlink privilege, and has never seen the cluster.

**Green gates prove the code is internally consistent. They do not prove it runs on the cluster.**

| Gate | Checks | Covers |
|---|---|---|
| `runner/test_phase1.sh` | 19 | registry loading, CLI validation, dry-run |
| `runner/test_phase2a.sh` | 4 | artifact markers, metadata, link fallback |
| `runner/test_phase3.sh` | 28 | sbatch generation, 3-layer override resolution, fan-out, marker durability |
| `runner/test_api_env.sh` | 21 | every task module imports without vLLM |
| `runner/test_loader.sh` | 20 | all eight parsers against frozen real-shaped fixtures |
| **Total** | **92** | |

Beyond the gates, all eight tasks were run end to end against **real paid APIs** (the `real10`
run), which is what surfaced bugs 1, 2, 5, 6, 7 and 8 — none of them were reachable with a mock
judge.

### Never executed anywhere

| Item | Why unproven |
|---|---|
| The `legacy` conda env | conda is not installed on the verification machine; `legacy.yml` has never been solved. |
| `neocoder` and `dat` on `legacy` | Both were only ever run by forcing the `api` env. Their vllm 0.5.3.post1 / torch 2.3.1 pin is untested. |
| `sbatch` accepting the generated directives | Generation is gated; no job has ever been submitted. Partition and account names are unverified guesses. |
| Any open-weight model | Every run to date used a hosted API model. |
| The `cygpath` branches in `_common.sh` on Linux | Guarded on `uname -s`, so they *should* be inert — but that branch has never executed there. |
| `creativity_index` end to end | Reaches evaluation, then blocks on the public infini-gram rate limit. |

---

## 6. Open items that need a decision

None of these are code problems. Each is a question about what the benchmark should report.

1. **`neocoder`'s comparison layer.** Faults 1 and 2 are mechanical and could be fixed in an
   afternoon. Fault 3 — multiple valid answers — requires per-problem checkers, which NeoCoder
   does not ship. Until then the reported correctness is a lower bound of unknown tightness.
2. **`creative_math`'s behaviour when a judge is unavailable.** Today it votes `NO`. The
   alternatives are to raise, or to record an abstention and require unanimity among the judges
   that answered. All three give different published numbers.
3. **The three unfixed Gemini wrappers.** Cheap to fix; nobody has needed them yet.
4. **`registry/models.yaml` still lists retired model IDs.** Of the eight API models registered,
   only `GPT4.1` and `GPT4.1-mini` still serve. See
   [RUNNING_GUIDE.md](RUNNING_GUIDE.md#c1--registering-the-model-you-want-to-test) — this is the
   single most likely first-run failure for a new user.
5. **Re-running anything published before these fixes.** At minimum, any `creativity_index`
   number (bug 4) and anything scored by `extract_yes_no` (bug 1).

---

## 7. Where the rest of the documentation is

| File | What it is |
|---|---|
| [RUNNING_GUIDE.md](RUNNING_GUIDE.md) | How to run it — three tracks, by audience. |
| [RESTRUCTURING_PLAN.md](RESTRUCTURING_PLAN.md) | The design contract and phase roadmap. Its "Read before run" section is the authoritative list of what is unverified. |
| [CHANGE_LOG.md](CHANGE_LOG.md) | Chronological record, with the commit ledger and per-phase deviations. |
| [WORKFLOW.md](WORKFLOW.md) | The original community-facing workflow description. Superseded operationally by the running guide. |
