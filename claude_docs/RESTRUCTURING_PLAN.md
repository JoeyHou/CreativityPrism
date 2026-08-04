# CreativityPrism Restructuring Plan

This document is the source of truth for the ongoing effort to restructure the CreativityPrism benchmark codebase. It captures the design rationale, the agreed architecture, and a phased implementation roadmap. New Claude Code sessions working on this project should read this file first.

---

## Read before run

Everything below was built and verified on one Windows laptop against hosted APIs. That machine
cannot build a conda environment, cannot create a symlink, and has never seen the cluster. The
gates are green on it, which proves the code is internally consistent — **it does not prove the
code runs on the cluster.** This section collects every "we skipped this because of the current
environment" note that is otherwise scattered across three documents, so the first person with
cluster access knows exactly what is still unproven.

### The environment everything was verified on

| | |
|---|---|
| OS / shell | Windows, Git Bash (MINGW64). No WSL. |
| conda | **Not installed.** The `modern` and `legacy` environments are conda specs that have never been instantiated here. |
| Python actually used | System Python 3.12.10 (no pandas) + a plain venv at `C:/cp-envs/creativityprism-api` (the `api` env, has pandas). |
| GPU / vLLM | None. No open-weight model has ever been loaded. |
| Symlink privilege | **Absent** (`WinError 1314`). Developer Mode is off. |
| SLURM | No `sbatch`, no `squeue`, no cluster login. |
| Network | Corporate. `api.deepseek.com` fails TLS handshake; OpenAI / Anthropic / Google all reachable. |

### Never executed anywhere — verify these first on the cluster

| Item | Why it is unproven | Where it lives |
|---|---|---|
| The `legacy` conda env | conda is not installed here, so `legacy.yml` has never been solved. There is deliberately no `legacy.txt`, because a `conda list --export` snapshot cannot be authored for an environment that has never been built. | `registry/environments/` |
| `neocoder` and `dat` on `legacy` | Both were only ever run by forcing the `api` env. Their pinned vllm 0.5.3.post1 / torch 2.3.1 stack is untested. | `registry/tasks/{neocoder,dat}.yaml` |
| `sbatch` accepting the generated directives | Script generation is gated (`test_phase3.sh`, 28/28), but no job has ever been submitted. Partition and account names are unverified guesses. | `runner/run.py`, `runs/*.yaml` |
| Whether the SLURM resource defaults are adequate | Never ran an open-model inference job. | `runs/*.yaml` |
| The `cygpath` branches in `_common.sh` | Guarded on `uname -s`, so they *should* be inert on Linux — but that branch has never been executed there. | `registry/adapters/_common.sh` |
| Any open-weight model | Every run to date used a hosted API model. | `registry/models.yaml` |

### Windows-only degradations (these disappear on the cluster)

- **`outputs/` contains `.path` reference files, not symlinks.** Without symlink privilege the
  runner writes a one-line `{kind}_output.path` file holding the native path. On Linux you get a
  real symlink and `readlink outputs/.../inference_output` works as documented.
- **Consequence for analysis code:** read `metadata.json` → `artifacts.{kind}.native_path`, which
  is populated identically on both platforms. Never assume a symlink exists. `result_analysis/loader.py`
  already does this; anything new should too.
- The `readlink` verification step in the Phase 2 checklist is therefore **impossible to run here**
  and remains unchecked, not failed.

### Live API facts as of 2026-08-03

Model IDs rot. These were probed directly with the repo's own keys on that date:

| Finding | Detail |
|---|---|
| Retired: `claude-3-7-sonnet-20250219`, `claude-3-haiku-20240307` | HTTP 404 `not_found_error`. The key is valid — a dead key returns 401, not a structured 404. |
| Retired: `gemini-2.0-flash`, `gemini-2.5-flash-lite` | HTTP 404 "no longer available". |
| **Listing ≠ serving** | `gemini-2.0-flash` still appears in `GET /v1beta/models` while refusing to serve. Do not trust the listing endpoint as an availability check; send a real one-token request. |
| **Gemini 2.5+ spends the output budget on thinking** | With a small `max_output_tokens` the response comes back `finishReason=MAX_TOKENS` with **no `parts` key at all** → `KeyError`/empty text. Fixed for the judge path by setting `thinking_config=ThinkingConfig(thinking_budget=0)`. |
| DeepSeek unreachable | `SSLV3_ALERT_HANDSHAKE_FAILURE` from this network. Not a key problem; not needed for judging. |

**Three Gemini wrappers still lack the thinking fix.** Only `tasks/math_n_index/api_warpper.py`
was corrected, because that is the judge path actually exercised. If you point a Gemini model at
another task, fix it there first or it will silently return empty text:

| File | Line | Status |
|---|---|---|
| `tasks/math_n_index/api_warpper.py` | ~79 | **Fixed** (`thinking_budget=0`) |
| `tasks/aut_ttcw_cshort/src/utils/api_wrapper.py` | ~75 | Not fixed |
| `tasks/ttct/src/utils/api_wrapper.py` | ~55 | Not fixed |
| `tasks/neocoder_dat/src/models/model.py` (`GenAIModel`) | ~268 | Not fixed |

### Scoring landmines — a dead judge does not look like an error

In `tasks/math_n_index/src/evaluation/creative_math_eval_api.py` and `api_eval.py`:

- Three of the four wrappers **return the exception as a string** instead of raising. That string
  is then compared against `"YES"`, so **any API failure is recorded as a `NO` vote.** A real
  artifact in this repo shows it happening: a Gemini `API_KEY_INVALID` message was stored as
  `"gemini-2.0-flash": "NO"` with the error text sitting in `reasons`.
- Correctness requires **unanimity** (`all(d == "YES" ...)`), so **one dead provider silently
  drives correctness to 0%** — indistinguishable from a genuinely wrong solution.
- Failures then **cascade**: correctness `NO` forces coarse novelty `NO`, which forces fine
  novelty `NO`. One retired model ID can zero an entire run.
- The judge panel is the hardcoded `JUDGE_MODELS` dict, **not** the config JSON. Renaming a judge
  requires editing both, or `get_api_key()` returns `None`.

**Fixed 2026-08-03 — `extract_yes_no()` used to match substrings.** It upper-cased the reply and
asked `if "YES" in text ... elif "NO" in text`, which meant:

- `"NO"` is a substring of **`NOT`, `CANNOT`, `KNOW`, `NOTE`, `NOVEL`**. A judge that opened with
  "I need to know whether…" was recorded as a `NO` vote. Observed live on 2026-08-03.
- `"YES"` was tested first and anywhere in the string, so `"The answer is not YES"` scored `YES`.

All three copies (`api_eval.py`, `src/utils/utils.py`, `src/utils/helpers.py`) now take the first
**whole-word** `YES`/`NO`, case-insensitively — every prompt asks for the verdict first and the
explanation after. Each copy keeps its original fallback, so `api_eval.py` still returns `UNCLEAR`
and the vLLM copies still return `NO`. Net effect on published numbers: a false `YES` caused by the
word appearing later in an explanation now resolves to the real verdict, and a reply with no verdict
token at all becomes `UNCLEAR`, which every downstream tally already counts exactly like `NO`. The
vLLM copies additionally recognise a lower-case `yes`, which previously scored `NO`.

### `creativity_index` depends on a public API that rate-limits hard

`evaluation_creative_index_parr.py` resolves every n-gram against `https://api.infini-gram.io/`,
a free public endpoint with no key. Measured from this machine on 2026-08-03 it answered a plain
`403 {"message":"Forbidden"}` for **roughly half of all requests, even at one request per second**
— it is not a payload problem (the legacy `corpus`/`engine` fields still work alongside the
documented `index` field) and not a header problem (curl and `requests` fail at the same rate).
The default `--num_workers 8` with a 0.1 s throttle makes it far worse.

Two things followed from that:

- The retry loop used a **flat 0.2 s delay over 5 attempts**, i.e. every attempt landed inside the
  same throttle window. It now uses 8 attempts with exponential backoff and jitter.
- On exhaustion it fell through to `occurrence = 0`, which is **the same value as "this n-gram is
  not in the corpus"**. Lost lookups therefore lowered coverage, and lower coverage *raises* the
  reported creativity index — a run degraded silently and in a flattering direction. It now raises
  instead. **Any previously published `creativity_index` number computed while the API was
  throttled is suspect.**

`creativity_index` inference works locally; its evaluation could not be completed here.

### The loader's `creative_math` join key collided (fixed)

`_parse_creative_math` joined inference to evaluation on `(problem_id, question_number)`. The same
problem is asked several times with different `k` (the number of reference solutions shown), so an
18-record run collapsed to **10 distinct keys** and every collision inherited the last record's
verdicts. Measured effect: the loader reported correctness `27.78%` where the evaluation script
printed `55.56%`. The key now includes `k` and the two agree exactly. If you have notebooks or
figures built from a loader run older than 2026-08-03, regenerate them.

### `ttct` only ever scores the `cot` variant

`ttct_evaluation.py` hardcodes `basic_eval_pred = ["SKIPPED" for _ in data]` and the same for
`instructive`, in **both** the vLLM and the API branch, each under a `# TODO: change this`. The
real judge call is made only for `cot`. So a ttct run used to pay for three generations per item
and score one.

**Fixed 2026-08-03, by defaulting rather than deleting.** `ttct_inference.py` already accepts
`-prompt_formats` and already fills a variant with `SKIPPED` when it is not requested; the flag
just defaulted to `all` and the adapter never passed it. The default is now `cot`, and
`registry/adapters/ttct.sh` passes `-prompt_formats cot` as well so the pipeline stays cot-only
even if that default moves again. Deleting the two variants outright was considered and rejected:
`basefile.csv` and the `csv2json`/`json2csv` round trip in `ttct_evaluation.py` (lines 33-38 and
55-60) address six `infer_*` columns by name, and the 45 committed judge-output files under
`human_annotation/data/mturk_anno/ttct/v1.1_llmj_output/` hold **real** basic and instructive
generations (675 of each) that the agreement analysis is built on. The flag reaches the same
outcome without touching either.

**A default ttct run is 500 items, not 700.** `basefile.csv` holds 7 question types x 100, but
`DEFAULT_SUBSET` scores 5 of them -- the judge rubric was human-aligned for those five only, and
`3_just_suppose` and `7_story` ship unscored. Their 200 rows stay in the file, carrying `SKIPPED`,
so the csv schema and the row-count assertion below stay fixed. 500 items x 1 variant is now the
full cost of a ttct run; it used to be 1500 generations.

Note that `assert len(csv_data) == len(input_data)` at line 28 compares **row** counts against
`basefile.csv`'s 700. The three variants are columns of one row, so neither the flag nor a
deletion would trip it -- do not treat that assertion as a guard on this.

One related note: the inference script marks surplus in-subset rows with `skip`, but rows
**outside** the configured subset are never marked -- their prompts are the literal string
`SKIPPED`. The loader drops any variant whose output is that sentinel (a 10-item run went from
630 rows, 600 of them placeholders, to 30).

`-num_samples` is **per question type**, not a total. It used to be multiplied by the three
prompt formats; with `-prompt_formats cot` a `--limit 2` over 5 question types is 10 items and
10 generations, not 30.

### `ttct` now reports the four TTCT traits

Resolved on 2026-08-03. ttct previously contributed **no numeric score at all**, because
`_parse_ttct` filled `eval_score` only when the judge's answer was itself a number and the ttct
judge answers in prose ending with a rubric block:

```
### Scores ###
Fluency: 5
Flexibility: 5
Originality: 4
Elaboration: 4
```

The parser for that block already existed — it just lived in a notebook, `extract_scores` in
`human_annotation/notebooks/mturk_agreement.ipynb`, and had never been ported. `loader.py` now
carries it verbatim as `_parse_ttct_scores`, down to the `split("### Scores ###")[1]` that reads
the text after the *first* marker, so the loader and the published agreement numbers cannot
disagree. A scored variant expands into four rows (`fluency`, `flexibility`, `originality`,
`elaboration`) the way `ttcw` expands into one row per rubric question; averaging them is a
reporting decision and stays out of the loader. Verified on 90 judge responses across the six
`zero_shot` corpora in `human_annotation/data/` — 4/4 traits parsed on every one — and the
`real10` run went from 0 numeric scores to 40.

### Token limits: the adapter's `max_new_tokens` was silently ignored for API models

Fixed on 2026-08-03, but worth understanding because the failure was invisible:
`ModelWrapper` in `tasks/math_n_index/api_warpper.py` hardcoded `max_tokens=30` for all
providers. The vLLM path honoured the configured `max_new_tokens`, the API path did not — so
`CREATIVITYPRISM_MATH_MAX_NEW_TOKENS=2000` had no effect and **every API generation was a
~150-character stub**. The judges then correctly reported the solutions as "incomplete" and the
run scored 0%. `ModelWrapper` now takes `max_tokens`; `inference_driver.py` passes the configured
value, and judges fall back to `CREATIVITYPRISM_MATH_API_MAX_TOKENS` (default 512, set it to 30
to reproduce the old truncated numbers).

**Any published `creative_math` number produced by an API model before this date was computed
from 30-token generations.**

### Windows console encoding

Model output routinely contains characters outside the Windows ANSI code page (a judge writing
`✓` is enough). Python then defaults stdout to cp1252 and a mid-run `print` raises
`UnicodeEncodeError`, aborting an evaluation that had already made every paid call. The runner
now exports `PYTHONIOENCODING=utf-8` to adapters and decodes their stdout as UTF-8. Inert on
Linux, where UTF-8 is already the default.

### A silent task is usually a buffered task

The adapter's stdout is a pipe, so Python block-buffers it. `ttct` ran for over an hour showing
literally nothing — not even its first `print()` — and looked hung. The runner now also exports
`PYTHONUNBUFFERED=1`, so progress appears live. If you wrap a long run in `cmd | tail -N`,
expect the same symptom from the shell side: `tail` holds *all* output until the command exits.
Redirect to a log file and `tail` the file instead.


### neocoder executes generated code, and that used to poison both liveness and scores

Two defects met here, and neither announced itself.

The runner used to hand its own stdin to the adapter. Codeforces solutions read stdin, so attached
to a terminal the generated code blocked on a prompt nobody was watching. Every one of those calls
hit the harness's 6-second limit: **55 of 60** generations were recorded as `code execution
timeout`. Re-running the identical artifacts with stdin closed gives **0 of 60**. The runner now
passes `stdin=subprocess.DEVNULL`, which is what a batch scheduler supplies anyway — but if you
ever invoke an adapter by hand, close its stdin or you will publish a near-zero correctness score
that has nothing to do with the model.

The harness's timeout helper also started **non-daemon** threads. Python cannot kill a thread, so a
timed-out one is merely abandoned — and the interpreter's shutdown handler then waits for it. The
process therefore hung *after* its results were written: correctness saved at 09:25 and the process
was still alive and idle 20 minutes later. Fixed with `thread.daemon = True`, which changes no
score.

**Still open, and it is a scoring decision — but no longer a mystery.** With both fixed,
`neocoder` correctness is still `0/60`. Instrumenting the bare `except:` in `test_correctness`
against the existing `real10` artifacts (no API calls) named the cause exactly:

```
=== exec / parse stage ===            (nothing: all 60 parse, exec and define solve)
=== first try (all cases at once) ===
    54  IndexError: list index out of range
     4  ran but mismatched
     1  SyntaxError: invalid syntax
     1  AttributeError: module 'sys' has no attribute 'input'
=== second try (one case at a time) ===
   256  IndexError: list index out of range
```

`parse_code` is not the culprit and neither is `exec`. **55 of the 60 generations** begin like this:

```python
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
```

That binds a *local* `input` to the real stdin reader, so `mock_input()`'s patch of
`builtins.input` is never consulted. `parse_code` does try to rewrite stdin reads, but every one
of its six rules requires parentheses — `sys.stdin.read()`, `.readline()`, `.readlines()` — and
the idiom above has none. Under the old inherited terminal stdin those 55 blocked forever, which
is precisely where the 55 "code execution timeout" verdicts came from; with stdin at `/dev/null`
they instead read `''`, leaving `data` empty so the first subscript raises. Only the 5 generations
that call plain `input()` have ever actually been evaluated by this harness.

Two ways out, both moving published numbers:

- **Extend the rewrite.** Catch the parenthesis-free assignment as well. Awkward, because
  `sys.stdin.read` returns the *whole* input in one call while `mock_input` yields one line at a
  time, so the replacement has to be a closure over the remaining lines rather than `input`.
- **Run the solution in a subprocess with the test case on real stdin.** This is what Codeforces
  itself does, it makes `input()`, `sys.stdin.read` and `sys.stdin.readline` all behave, and it
  deletes the whole monkeypatching layer along with its timeout-thread problems.

**The second was implemented on 2026-08-03.** `evaluation_utils.run_solution` writes the parsed
code to a temp directory, runs it under `sys.executable` with the case piped to real stdin, and
enforces a timeout the OS can actually act on. `test_correctness` no longer `exec`s into the
evaluator's own globals — which also removes the `del globals()['solve']` cleanup and the risk of
one problem's `solve` leaking into the next — and detects unrunnable code with `compile()` instead.
Measured against the same `real10` artifacts: **"code not executable" went from 60 of 60 to 0 of
60.** Every generation now runs.

### neocoder's *comparison* layer is still wrong, and that is a separate decision

With execution fixed, `neocoder` scores 3 of 60 rather than 0 of 60. The remaining gap is not the
runner and not the models. Three distinct problems, in the order they bite:

1. **The first try cannot ever compare.** It flattens every case's output into one list of lines
   and zips it against `test_case_outputs`, which is a list *per case*. So
   `type_agnostic_compare('9', ['9', '10 1', '15 5'])` runs `eval('9')` and then
   `list(map(str, 9))`, which raises. The first try therefore always falls through to the retry.
2. **The per-case retry feeds a malformed file.** A stored case such as
   `[['2'], ['15', '1', '10', '5']]` is `n` followed by the data — it carries no leading test
   count, because the harness is supposed to supply one. The first try does
   (`test_input.insert(0, str(num_test_cases))`); the retry does not. Solutions written for
   Codeforces multi-test input read the `n` line as `t` and then crash.
3. **Even corrected, it exact-matches a single reference answer.** Feeding the retry a `1` header
   and comparing line by line still yields 3 passes, 4 crashes and 53 "differs from reference".
   The one case inspected in full, `1895B`, is a *correct* answer: the model printed the same
   optimal cost and a different but equally optimal pairing (`1 10` where the reference has
   `10 1`). Codeforces scores that class of problem with a per-problem checker, which NeoCoder
   does not ship. How many of the other 52 are genuine failures is unmeasured — do not assume.

Fixing 1 and 2 is mechanical. Fixing 3 is a benchmark design question. Neither was done, because
both move published numbers and neither is a runner concern.

Two smaller oddities from the same artifacts: `new_techniques_ratio` is `1.0` for all 55 scored
rows, and 5 of 60 generations never reach the creativity CSV because the upstream evaluator drops
them. Decide what `neocoder` should report before treating its numbers as publishable.


### Diagnostics must not be able to end a run

The `infini-gram` retry handler printed from eight worker threads into one redirected stdout. On
Windows that raised `OSError: [Errno 22]` *inside the exception handler*, killing a 40-minute
evaluation. The print is now wrapped so it cannot propagate. Worth copying that habit: anything
logged from a worker thread during a paid run should be unable to abort the run.


### Gate gotcha

Run the gates in a shell where these are unset, or they will inherit a forced env / a dead mock
endpoint from a previous session and fail confusingly:

```bash
unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL
bash runner/test_phase1.sh && bash runner/test_phase2a.sh && bash runner/test_phase3.sh
bash runner/test_api_env.sh && bash runner/test_loader.sh
```

---

## Current Development Checkpoint (2026-08-01)

### Canonical workspace and Git state

- Continue restructuring work in the clean worktree whose directory basename is **`creativityprism_v2-mainv2-clean`**.
- Its local branch is **`main_v2_publish`**, tracking **`personal/main_v2`**.
- The verified Phase 1 checkpoint is commit **`56cfbd3ce564535a2416cb847641660da2a70118`** (`Add registry-driven Phase 1 runner`).
- The verified Phase 2A + 2B checkpoint is commit **`747a5ba1dcc4848410d807f921991bd93d122fb9`** (`Add artifact contract, centralized outputs, and evaluation dispatch`).
- The `personal` remote uses `git@github-personal:JoeyHou/CreativityPrism.git`. The existing HTTPS `origin` is preserved and must not be rewritten as part of restructuring work.
- Public `main` was not modified by either publication; both were pushed to the separate remote branch `main_v2`.

### Publishing procedure

**Push from the VS Code Git UI (Source Control view), not from an agent-run terminal.** The `personal` remote is SSH with a passphrase-protected key, and `git push`/`git fetch` from a terminal blocks on an invisible `Enter passphrase` prompt; the VS Code UI surfaces it as a dialog instead. An agent must therefore stop at the commit and hand the push over.

Verify a push without touching SSH — the local tracking ref is updated by a successful UI push, and the public repo answers `ls-remote` anonymously over HTTPS:

```bash
git rev-parse HEAD personal/main_v2          # must match
git rev-list --left-right --count HEAD...personal/main_v2   # must be 0  0
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/JoeyHou/CreativityPrism.git 'refs/heads/*'
```

The last command is the authoritative check: `refs/heads/main_v2` must equal local `HEAD`, and `refs/heads/main` must still be `4705a830501e47b999481a0ec0c62ac2cca10c86`.

On Joey's current Windows machine, the parent development directory contains:

```text
creativityprism_v2/
├── .venv/                         # local environment; never commit
├── creativityprism_v2/            # old/dirty worktree; read-only reference
└── creativityprism_v2-mainv2-clean/ # canonical restructuring worktree
```

The old `creativityprism_v2/` worktree mixes ZIP-derived missing files, remote history, and local changes. **Do not develop, stage, reset, clean, or push from it.** Keep it as a read-only migration source until its remaining research assets have been explicitly audited. Do not delete it yet.

### New-session startup gate

After opening `creativityprism_v2-mainv2-clean` as the VS Code workspace root, run:

```bash
git status --short --branch
git rev-parse HEAD
bash runner/test_phase1.sh
bash runner/test_phase2a.sh
bash runner/test_api_env.sh   # skips unless the api venv exists
```

Expected baseline:

```text
branch/upstream: main_v2_publish...personal/main_v2
HEAD:            747a5ba1dcc4848410d807f921991bd93d122fb9, or a later docs-only commit on top of it
worktree:        clean
Phase 1 gate:    19 passed, 0 failed
Phase 2A gate:   4 passed, 0 failed (33 unit tests)
```

If the branch has advanced normally, the exact HEAD may differ; require a clean worktree, review intervening commits, and rerun the gate before editing. Never reset a newer valid branch merely to reproduce the checkpoint hash.

### Next implementation slice

**Phase 2A (artifact contract), 2B (evaluation dispatch), and 2C (all remaining task
adapters) are complete.** All eight tasks now run through `runner/run.py`. Nothing has
been executed on the cluster yet, so the next slice is verification, not more wiring:

1. Build the `legacy` env (`bash scripts/setup_envs.sh --env legacy`) and capture a
   `registry/environments/legacy.txt` snapshot from the result.
2. Download GloVe into `tasks/neocoder_dat/embeddings/glove/` (needed only by `dat` eval).
3. Run the Phase 2 end-to-end verification block below, including a small-`--limit`
   eval run against a real judge.

### Task adapters (Phase 2C)

| Task | Bundle | Env | `--limit` | `--judge-model` |
|------|--------|-----|-----------|-----------------|
| `neocoder` | `tasks/neocoder_dat` | `legacy` | not supported | **no-op** — technique detection hardcodes `gpt-4-turbo` |
| `dat` | `tasks/neocoder_dat` | `legacy` | maps to `--repeat` | **no-op** — GloVe distance, no LLM judge |
| `creative_math` | `tasks/math_n_index` | `modern` | supported | **no-op** — fixed 3-judge majority vote |
| `creativity_index` | `tasks/math_n_index` | `modern` | supported | **no-op** — n-gram metric, no LLM judge |

Constraints discovered while wiring these, all verified in the task sources:

- **`creative_math_eval_api.py` loads its config at import time** from a fixed relative
  path, so the adapter cannot pass one as an argument. It now reads
  `CREATIVITYPRISM_MATH_EVAL_CONFIG` first. That script also wants a **flat** config
  (`config["file_paths"]`), whereas the checked-in `configs/eval_creative_math.json`
  is `experiments_list`-shaped and would raise `KeyError` — another reason the adapter
  generates its own.
- **The two creative_math paths agree only by convention.** Inference writes
  `<generation>/<alias>.json`; eval rebuilds `<generation>/<alias>/<alias>.json`. The
  adapter therefore hands inference `.../creative_math/<alias>` and eval the parent
  `.../creative_math`.
- **`creativity_index` eval `--subset` defaults to 100 and silently truncates.** The
  adapter always passes it explicitly: `--limit` when given, otherwise the generation
  file's real length.
- **`creative_index_inference.py` capped every run at `data[:100]`** regardless of
  `portion`. Removed; both math drivers now honour an exact `test_size`, which is how
  the runner's integer `--limit` is expressed (converting it to a float `portion`
  would be lossy).
- **NeoCoder evaluation is a strict three-step chain**: `correctness` -> `detection` ->
  `creativity`, because `calculate_creativity()` asserts that both `correctness` and
  `techniques` have been written back into the inference file. `correctness` and
  `creativity` also write the *same* `<model>_sample=..._creativity.json` filename, so
  the adapter gives them separate save folders.
- **NeoCoder `correctness` executes model-generated Python** and `detection` bills one
  `gpt-4-turbo` call per generated solution. Both are inherent to the benchmark; the
  adapter documents them in its header.
- **`dp_rounds` must match between inference and scoring.** `calculate_creativity()`
  defaults to 5 while `inference_dp.py` defaults to 3, so the adapter pins 5.
- **Two adapters announce a directory, not a file.** NeoCoder's filename embeds a
  sample count only the task can compute, and `creativity_index` produces one file per
  domain. Both directories are run-scoped, so the artifact stays unambiguous.
- **`evaluate_dat.py` had no output-path option**; it derived one by replacing the
  substring `inference` with `evaluation` inside the input path, which **overwrites the
  inference file** when no such segment exists. It now takes `--output-path`, refuses to
  write over its input, and the adapter always passes the flag.
- **DAT's GloVe and word-list paths were hardcoded to a site-local AFS location.** They
  now resolve from flags, then env vars, then `embeddings/glove/glove.840B.300d.txt` and
  the bundled `words.txt`. GloVe is ~2 GB and is not redistributed; `embeddings/` is
  gitignored and the missing-file error prints download instructions.
- **The root `.gitignore` pattern `*/data/outputs/*` matches only one path segment**
  before `data/`, so it never covered `tasks/math_n_index/`. Explicit entries were added;
  `tasks/neocoder_dat/results/` was already covered by that bundle's own `.gitignore`.

#### Provider API keys

`tasks/neocoder_dat` reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GENAI_API_KEY` /
`DEEPSEEK_API_KEY` straight from the environment, so `tasks/math_n_index` was patched
to do the same instead of reading keys out of its JSON configs. `_common.sh` gained
`export_provider_keys`, which runs `registry/adapters/_provider_keys.py` against
`$CREATIVITYPRISM_API_KEYS` and evals the resulting shell-quoted `export` lines.
Keys already present in the environment win; placeholder values (`YOUR_...`) and empty
strings are skipped. `api_keys.json` stays model-keyed for the older tasks — the
provider block is purely additive.

### Evaluation dispatch (Phase 2B)

Each adapter's eval branch runs behind `[[ "$MODE" == "eval" || "$MODE" == "both" ]]` and emits `CP_ARTIFACT eval <path>` only after the eval command returns zero (`set -e` guarantees this).

| Task | Eval entry point | Judge alias key | Announced eval artifact |
|------|------------------|-----------------|--------------------------|
| aut | `run_evaluation.py <cfg>` with `task: aut_push` | `api_call`, else `aut_ttcw_cshort` | `data/output/<run_id>/aut/<alias>/eval_output_cleaned.json` |
| ttcw | `run_evaluation.py <cfg>` with `task: creative_writing` | `api_call`, else `aut_ttcw_cshort` | `data/output/<run_id>/ttcw/<alias>/eval_output_cleaned.json` |
| creative_short | `run_evaluation.py <cfg>` with `task: creative_short` | **none — automated metrics** | `data/output/<run_id>/creative_short/<alias>/eval_output_cleaned.json` |
| ttct | `src/evaluation/ttct_evaluation.py` | `ttct` | `data/evaluations/<run_id>/<model_short>.json` |

Key constraints discovered while wiring this, all verified in the task sources:

- **The eval config's `run_id` must equal the inference config's `run_id`.** The bundled evaluator hardcodes its input as `data/output/<run_id>/inference_output.json`; there is no path option. A gate test asserts each bundled adapter contains that `run_id` entry exactly twice.
- **In a bundled *eval* config, `model_name` is the judge**, not the model under test. The model under test is identified only by `run_id`.
- **The bundled evaluator writes back into the inference directory** (`eval_output_cleaned.json` plus `eval_report.csv`). The announced eval artifact is therefore the specific per-item file, not the directory, so that the `eval` link does not simply duplicate the `inference` link. `eval_report.csv` stays reachable one hop away through the inference link.
- **`creative_short` uses no LLM judge.** Its evaluation is fully automated (DSI, n-gram diversity, inverse homogenization, novelty, theme uniqueness). `--judge-model` is still required by the runner but has no effect. Its metric models (`thenlper/gte-large`, `en_core_web_sm`, `benepar_en3`) download lazily on first use.
- **`ttct_evaluation.py` defaults `-api_key_path` to a hardcoded `/ihome/xli/joh227/...` path**, so the adapter always passes `-api_key_path "$CREATIVITYPRISM_API_KEYS"`. Its `-temp` only selects a fallback input directory when `-run_id` is empty, so it is deliberately not forwarded; its `-summary`/`-demo`/`-pairwise` flags are `type=bool`, which in argparse makes any non-empty string truthy, so they are left at their defaults rather than passed explicitly.

#### Credentials: passed by path, never copied

`tasks/aut_ttcw_cshort/src/driver.py` loaded `./api_keys.json` relative to the task directory, while the runner exports `CREATIVITYPRISM_API_KEYS` pointing at repo root. Those are different files, so an API judge could never find keys. The fix mirrors the Phase 1 precedent in `ttct_inference.py`: prefer the env var when it names an existing file, otherwise fall back to `./api_keys.json`. This is deliberately non-regressive — a machine that only has the task-local file keeps working.

Adapters must **never** copy or symlink the credentials file into the repo tree. A gate test enforces this.

### Deferred cleanup backlog

Accumulated rough edges. None are blocking; do them as one focused slice **after** Phase 2 lands, so that a cleanup diff never mixes with a behavior diff.

| Item | Where | Why it is deferred |
|------|-------|--------------------|
| Inconsistent phase labels in the module docstring (`Phase 2.1a` vs `Phase 2A`) | `runner/run.py` header | Cosmetic, but confusing to new readers |
| `main()` does argument resolution, env setup, the task loop, and output materialization in one ~90-line function | `runner/run.py` | Should split into `run_one_task()` + `main()`; no behavior change intended, so it needs its own diff |
| `--run-id` retained as a deprecated alias for `--label` | `runner/run.py` | Remove once no local scripts or notebooks pass it; grep first |
| `try: from runner import artifacts / except ImportError: import artifacts` | `runner/run.py` | Needed because the file is used both as a script and as an importable module. Cleaner fix is a real `runner/__init__.py` plus `python -m runner.run`, which changes the documented CLI, so it needs a deliberate decision |
| Two separate gate scripts (`test_phase1.sh`, `test_phase2a.sh`) | `runner/` | Fine for now; consider one `runner/test_all.sh` wrapper once Phase 2 is complete |
| Six near-identical ephemeral-config heredocs (three adapters x inference/eval) | `registry/adapters/{aut,ttcw,creative_short}.sh` | Now clear what varies: `run_id`, `task`, `model_name`, and an optional `test_size`, so a `_common.sh` helper is feasible. Deferred because `test_phase1_behavior.py` asserts each adapter's source literally contains its own task-qualified `run_id` entry; hoisting the heredoc removes those literals, so the refactor must land with a rewritten Phase 1 assertion in its own diff. |
| `technique_detection()` writes `human_solution_techniques.json` back into the tracked dataset directory, and `creative_math_eval_api.py` appends to `evaluation_results_all_models.jsonl` in the CWD | `tasks/neocoder_dat/src/utils/configs.py`, `tasks/math_n_index/src/evaluation/creative_math_eval_api.py` | Both are outside run isolation. The first is idempotent in practice (the file is already populated and is only extended for unseen problem ids); the second is append-only. Fixing them means changing task-internal output contracts, so it needs its own diff. |
| `registry/environments/legacy.yml` has no `legacy.txt` counterpart | `registry/environments/` | `.txt` files are `conda list --export` snapshots; one can only be produced after the env has actually been built on the cluster. |

### Safety constraints carried forward

- Do not read, print, stage, or commit credential values. `api_keys.json` and site-local environment files remain ignored.
- `claude_docs/PERSONAL-GITHUB-PUSH-PROMPT.md` is local-only and must never enter Git history.
- Do not use `git add .`, force push, or push restructuring changes directly to public `main`.
- Preserve unrelated user changes and the old worktree until migration completeness is proven.
- Paid API smoke tests require explicit cost review; use deterministic tests and dry runs first.

---

## Goals

Two driving goals shape every decision in this plan:

1. **Easy to add new tasks** — including from the community. Adding a task should require dropping a folder and adding ~2 files in `registry/`. No central code modifications.
2. **Easy to analyze task outputs** — for downstream analysis and data inspection. All outputs should be discoverable from a single location with a consistent naming convention.

---

## Architecture

### Top-level layout (target state)

```
creativityprism_v2/
├── tasks/                    # Self-contained task implementations (unchanged by restructuring)
│   ├── aut_ttcw_cshort/      # Bundled folder hosting aut, ttcw, creative_short
│   ├── ttct/
│   ├── neocoder_dat/         # Bundled folder hosting neocoder, dat
│   └── math_n_index/         # Bundled folder hosting creative_math, creativity_index
│
├── registry/                 # ALL integration glue lives here
│   ├── tasks/                # One YAML per logical task
│   │   ├── aut.yaml
│   │   ├── ttcw.yaml
│   │   ├── creative_short.yaml
│   │   ├── ttct.yaml
│   │   ├── neocoder.yaml
│   │   ├── dat.yaml
│   │   ├── creative_math.yaml
│   │   └── creativity_index.yaml
│   ├── adapters/             # One shell adapter per logical task
│   │   ├── aut.sh
│   │   ├── ttcw.sh
│   │   ├── ...
│   ├── environments/         # Conda env requirements files
│   │   ├── modern.txt        # vllm 0.7.x, torch 2.5.1 — covers aut bundle, ttct, math_n_index
│   │   ├── legacy.txt        # vllm 0.5.3, torch 2.3.1 — covers neocoder_dat
│   │   └── .location         # gitignored: stores conda prefix path if custom
│   └── models.yaml           # Canonical model name → task-specific aliases
│
├── runner/                   # The unified CLI orchestrator
│   ├── run.py                # Main entry point
│   ├── slurm_template.sbatch # SLURM header template
│   └── test_phase1.sh        # Smoke tests (added in Phase 1)
│
├── scripts/
│   └── setup_envs.sh         # Creates conda envs from registry/environments/*.txt
│
├── outputs/                  # Centralized view of all task outputs (gitignored)
│   └── {label}/
│       └── {task}/
│           └── {canonical_model}/
│               ├── inference_output.json   # symlink to native location
│               ├── eval_output.json        # symlink
│               └── metadata.json           # generated by runner
│
├── result_analysis/          # Existing folder; will gain loader.py in Phase 3
│   ├── loader.py             # NEW (Phase 3): unified output loader
│   ├── output_length_analysis.ipynb
│   └── visualization.ipynb
│
└── RESTRUCTURING_PLAN.md     # This file
```

### Key design principles (do not re-litigate)

1. **Thin integration layer, not a rewrite.** Wrap existing task code; do not refactor internals. Existing `run_inference.py`, `src/`, etc. remain untouched.

2. **Self-contained registry.** Adding a new task requires touching only `registry/`:
   - `registry/tasks/{name}.yaml` — declarative task metadata
   - `registry/adapters/{name}.sh` — shell adapter that knows how to run the task
   - (and the actual task folder under `tasks/` if it's new)

3. **Per-task shell adapters, called via CLI args.** Each adapter:
   - Receives unified args: `--model`, `--run-id`, `--output-dir`, `--limit`, etc.
   - Translates the canonical model name to whatever the task expects (via `models.yaml`)
   - Calls the task's existing scripts
   - Prints `CP_ARTIFACT <kind> <native_path>` on stdout so the runner can link the result

4. **Centralized `outputs/` via links.** Tasks write to their native output location. After completion, the runner links them under `outputs/{label}/{task}/{canonical_model}/`. No double storage; direct task reruns still work.

5. **Canonical model names enforced at runner level.** Users always pass `GPT4.1`, never `gpt-4.1-2025-04-14`. `registry/models.yaml` is the single source of truth for translation.

6. **No metric definitions in registry (yet).** Post-processed metrics like `best_avg_rating` currently live in notebooks. We will formalize them later if needed.

7. **Runner is dumb.** It reads YAML, builds commands, runs them. Zero task-specific knowledge in `runner/run.py`.

8. **Per-environment conda envs (not per-task).** Two envs cover all existing tasks: `creativityprism-modern` (aut bundle, ttct, math_n_index — vllm 0.7.x) and `creativityprism-legacy` (neocoder_dat — vllm 0.5.3). Each task declares its env in its YAML; multiple tasks can share. Contributors who need a new env add a `registry/environments/{name}.txt` file. The runner pre-flight checks env existence and fails loudly with setup instructions if missing.

---

## File formats

### `registry/tasks/{name}.yaml`

```yaml
name: aut
display_name: "Alternative Uses Task"
domain: divergent_thinking
folder: tasks/aut_ttcw_cshort
adapter: registry/adapters/aut.sh
environment: modern              # references registry/environments/modern.txt
description: "Generate creative alternative uses for common objects"

# Optional: SLURM defaults for this task (Phase 3)
slurm:
  partition: gpu
  time: "2:00:00"
  mem: 32G
  gres: "gpu:1"
```

### `registry/models.yaml`

```yaml
models:
  GPT4.1:
    type: api
    provider: openai
    aliases:
      api_call: gpt-4.1-2025-04-14
      aut_ttcw_cshort: gpt_4.1
      ttct: gpt-4.1-2025-04-14
      neocoder_dat: gpt-4.1
      math_n_index: gpt-4.1

  Qwen2.5-72B:
    type: open
    provider: huggingface
    hf_id: Qwen/Qwen2.5-72B-Instruct
    aliases:
      aut_ttcw_cshort: qwen_72b_instruct
      ttct: Qwen2.5-72B-Instruct
      neocoder_dat: Qwen2.5-72B-Instruct
      math_n_index: Qwen2.5-72B-Instruct

  # ... seeded from MODEL_NAME_MAP in result_analysis/output_length_analysis.ipynb
```

### Adapter contract: `registry/adapters/{name}.sh`

Each adapter is a bash script invoked by the runner as:

```bash
bash registry/adapters/aut.sh \
    --model GPT4.1 \
  --judge-model GPT4.1-mini \
    --run-id v3 \
    --output-dir /abs/path/to/outputs/v3/aut/GPT4.1/ \
    [--limit 5] \
    [--inference-only|--eval-only]
```

The adapter must:
1. Activate its declared conda env at the top (the runner exports `CREATIVITYPRISM_ENV_PREFIX` if a custom prefix was set)
2. Parse those CLI args
3. Look up the task-specific model alias from `registry/models.yaml`
4. Invoke the task's existing inference (and/or evaluation) code
5. Announce each artifact it actually produced (see the artifact contract below)
6. Exit 0 on success, non-zero on failure

### Artifact contract (Phase 2A)

Adapters announce produced artifacts by printing marker lines on stdout:

```text
CP_ARTIFACT inference /abs/path/to/native/inference/output
CP_ARTIFACT eval /abs/path/to/native/eval/output
```

Use the `emit_artifact <kind> <path>` helper from `registry/adapters/_common.sh`.

Rules:

- Valid kinds are `inference` and `eval`. Unknown kinds are warned about and ignored.
- A marker must be emitted **only for a phase that actually ran and succeeded**. An `--eval-only` invocation must not announce an inference artifact.
- The path may be a file or a directory, and may contain spaces. The last marker of a kind wins.
- The Phase 1 marker `OUTPUT_PATH=<path>` is still parsed as an `inference` artifact for backward compatibility. An explicit `CP_ARTIFACT inference` marker overrides it.

The runner (`runner/artifacts.py`) then, per adapter invocation:

- Creates `outputs/{label}/{task}/{canonical_model}/`.
- Links each announced artifact as `{kind}_output` (directories) or `{kind}_output{suffix}` (files, e.g. `inference_output.json`). Symlink targets are relative for in-repo artifacts and absolute otherwise.
- Falls back to a one-line `{kind}_output.path` **reference file** when symlinks are unavailable (Windows without Developer Mode, exotic filesystems). Native outputs are never copied.
- Writes `metadata.json` (label, task, models, limit, mode, environment, adapter, command, exit code, timestamps, and one record per artifact). Artifact records accumulate across runs of the same triple, so an `--eval-only` rerun does not drop the earlier inference record.
- Never fails the run on a contract problem: a missing/nonexistent artifact is warned about on stderr and recorded with `"exists": false`. Adapter exit codes remain the only failure signal.

#### Platform behavior: symlink vs. `.path` reference

| Platform | What gets created | `link_type` in metadata |
|----------|-------------------|-------------------------|
| Linux / macOS (including the Pitt CRC cluster) | A real symlink, e.g. `inference_output -> ../../../../tasks/aut_ttcw_cshort/data/output/v3/aut/gpt_4.1` | `symlink` |
| Windows **with** Developer Mode or admin | A real symlink | `symlink` |
| Windows **without** Developer Mode | A one-line text file `inference_output.path` containing the native absolute path | `reference` |

**The `.path` fallback is a Windows-only degradation, not the normal case.** On the cluster you will always get real symlinks, and `readlink outputs/.../inference_output` behaves as the verification block below expects. The fallback exists so that a dev machine without symlink privilege (`WinError 1314`) can still run the pipeline instead of crashing.

Analysis code should therefore **read `metadata.json` rather than assume a symlink exists** — `artifacts.{kind}.native_path` is always populated on both platforms. This is what `result_analysis/loader.py` will do in Phase 3.

#### Marker durability (resolved in Phase 3, 2026-08-02)

Markers were **transient**. The runner captured them from the adapter's stdout pipe while the process was alive; nothing wrote the marker lines themselves to disk. That was safe locally because:

- Nested processes inherit the adapter's stdout, so a marker emitted by a helper script or a nested `bash` still reaches the runner's pipe.
- The durable record is `metadata.json`, written immediately after the adapter exits.

It broke under **SLURM submission**, where the adapter runs detached inside a batch job and its stdout goes to a SLURM log file the runner never reads. Two fixes were expected here — parse the job's stdout log after completion, or append markers to a file in the run's output directory.

**What was actually done takes the problem off the table instead.** The generated sbatch script re-invokes `runner/run.py` (minus the SLURM flags) rather than the adapter, so the runner runs *inside* the job and captures stdout exactly as it does locally. There is one code path, not two, and no post-hoc log parsing.

The marker file was implemented anyway as a second line of defence: `emit_artifact` appends to `$OUTPUT_DIR/.cp_artifacts`, and `materialize_run_outputs` merges that with stdout, **preferring stdout** because a stale sidecar can survive into a re-run of the same `(label, task, model)` directory.

### Environment management: `scripts/setup_envs.sh`

```bash
# Default: install envs in conda's default location (~/.conda/envs/)
bash scripts/setup_envs.sh

# Custom location (e.g., external storage to avoid filling home dir)
bash scripts/setup_envs.sh --prefix /external/storage/conda_envs
# or via env var:
CREATIVITYPRISM_CONDA_PREFIX=/external/storage/conda_envs bash scripts/setup_envs.sh

# Set up a single env only
bash scripts/setup_envs.sh --env modern

# Re-create from scratch
bash scripts/setup_envs.sh --env modern --force
```

Behavior:
- Idempotent: skips envs that already exist (unless `--force`)
- When `--prefix` is set, creates envs at `{prefix}/creativityprism-{name}` and writes the prefix to `registry/environments/.location` (gitignored)
- Adapters read `registry/environments/.location` (if present) to determine env location; otherwise use named envs in conda's default location
- The runner does a pre-flight check before invoking any adapter; if the env is missing it prints: `Environment 'creativityprism-modern' not found. Run: bash scripts/setup_envs.sh --env modern`

**Joey's machine**: install conda envs at `/ix1/xli/joh227/conda_envs/` (external storage, not home). Phase 1 should pre-create `registry/environments/.location` containing this path so `--prefix` doesn't need to be passed manually. The `.location` file is gitignored so other users/contributors set their own.

### Runner CLI

```bash
# Listing
python runner/run.py --list-tasks
python runner/run.py --list-models

# Dry-run (print command, no execution)
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --dry-run

# Run inference + evaluation
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --inference-only
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --eval-only

# Run all tasks
python runner/run.py --task all --model GPT4.1 --judge-model GPT4.1-mini --label v3

# Smoke test with limited samples
python runner/run.py --task aut --model GPT4.1-mini --judge-model GPT4.1-mini --label smoke_test --limit 5

# SLURM submission (Phase 3)
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --slurm
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --slurm --no-submit
```

---

## Phased Roadmap

### Phase 1: Foundation (Inference + Registry + Cleanup)

**Status:** Complete + Behavior-gated (2026-07-22; original API/GPU smoke tests 2026-04-12)

**Deviations from plan:**
- `--limit` is wired end-to-end for all four Phase 1 tasks. Public runner inputs must be positive integers; omitting the argument means full data. Task-internal `-1` defaults remain implementation details. All four YAMLs carry `limit_supported: true`.
- `tasks/ttct/src/inference/ttct_inference.py` was patched to accept `-run_id` so its output path mirrors `tasks/aut_ttcw_cshort` and matches what `ttct_evaluation.py` already expects (`data/outputs/<run_id>/<model>.json`). This is a small task-internal change but unifies output/eval paths under one run_id, which the user explicitly requested.
- Adapters share a `registry/adapters/_common.sh` helper for arg parsing, conda activation, and `models.yaml` lookup.
- Adapters generate ephemeral one-shot configs (instead of reading the existing per-model config files under `tasks/aut_ttcw_cshort/configs/`); `registry/models.yaml` is the sole source of truth for both the folder alias and the API model id.
- `registry/environments/.location` was pre-created with `/ix1/xli/joh227/conda_envs` and added to `.gitignore`.

**Post-completion fixes (2026-04-12) — found during end-to-end smoke testing:**

All four Phase 1 tasks were smoke-tested with `GPT4.1-mini` (API path) and AUT was additionally tested with `Qwen2.5-7B` (vLLM path). Four bugs were found and fixed:

1. **`--limit` not wired through adapters.** The runner accepted `--limit N` and passed it to adapters, but adapters ignored it. Fix: `aut.sh`, `ttcw.sh`, `creative_short.sh` inject `"test_size": N` into the ephemeral JSON config; `ttct_inference.py` accepts `-num_samples N`, and `ttct.sh` passes it.

2. **`PYTHONPATH: unbound variable` in `ttct.sh`.** `_common.sh` uses `set -u`; the line `export PYTHONPATH="$PYTHONPATH:$(pwd)"` blows up when `PYTHONPATH` is not already set. Fix: changed to `export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"`.

3. **`transformers 5.5.3` incompatible with vLLM 0.7.2.** pip resolved `transformers>=4.49.0` to 5.5.3, which broke vLLM's tokenizer init (`Qwen2Tokenizer has no attribute all_special_tokens_extended`). Fix: pinned `transformers>=4.49.0,<5.0.0` in `registry/environments/modern.yml`. The existing env was fixed in-place with `pip install "transformers<5.0.0"` (resolved to 4.57.6).

4. **Triton/vLLM cache writes to `/ihome` hit disk quota.** vLLM's Triton kernel compilation cache and usage-stats file default to paths under `/ihome`, which is quota-limited. Fix: `_common.sh`'s `activate_env()` now exports `TRITON_CACHE_DIR` and `TORCH_HOME` to `<conda_prefix>/.cache/{triton,torch}` (same scratch space as the envs), and sets `VLLM_NO_USAGE_STATS=1` to suppress telemetry writes.

**Behavior hardening (2026-07-22):**

5. **Bundled native output collision.** AUT, TTCW, and Creative Short used the same `data/output/<run_id>/<alias>` namespace. Their adapters now include the logical task: `data/output/<run_id>/<task>/<alias>`.

6. **Exact-limit off-by-one.** The bundled prompt producers stopped only after producing `N+1` prompts. Their stop conditions now use `>=`; TTCT's aligned slices were also behavior-tested.

7. **Undefined public non-positive limits.** The shared runner validation now rejects `0` and all negative values from both CLI and config submissions. Omitting `limit` requests the full dataset.

8. **Behavior gate.** `runner/test_phase1_behavior.py` executes the real prompt-selection paths with lightweight dependency stubs. Together with dry-run integration checks, `bash runner/test_phase1.sh` passes 19/19 checks.

**Smoke test results (run-id: `smoke_phase1`, 2026-04-12):**

| Task | Model | Type | Items | Result |
|------|-------|------|-------|--------|
| aut | GPT4.1-mini | API | 63 (full) | PASS |
| ttcw | GPT4.1-mini | API | 4 (requested limit 3) | Exposed off-by-one; fixed 2026-07-22 |
| creative_short | GPT4.1-mini | API | 4 (requested limit 3) | Exposed off-by-one; fixed 2026-07-22 |
| ttct | GPT4.1-mini | API | 3×3 formats (limit 3) | PASS |
| aut | Qwen2.5-7B | vLLM (4×L40S) | 4 items×5 rounds | PASS |

These historical API/GPU runs were not rerun on 2026-07-22. Current adapters print task-qualified `OUTPUT_PATH` values, and the deterministic gate verifies the path contract without invoking models or paid APIs.

**Scope:**
- Build `registry/` with YAMLs and adapters for **AUT bundle (aut, ttcw, creative_short)** and **TTCT** only.
- Build `runner/run.py` with inference support (no eval yet, no SLURM yet, no symlinking yet).
- Remove cs4 references from `generate.py`, `main/tasks/__init__.py`, and any shell scripts under `tasks/aut_ttcw_cshort/scripts/` that reference cs4.

**Files to create:**
- `registry/tasks/aut.yaml`
- `registry/tasks/ttcw.yaml`
- `registry/tasks/creative_short.yaml`
- `registry/tasks/ttct.yaml`
- `registry/adapters/aut.sh`
- `registry/adapters/ttcw.sh`
- `registry/adapters/creative_short.sh`
- `registry/adapters/ttct.sh`
- `registry/environments/modern.txt` (consolidated from `tasks/aut_ttcw_cshort/requirements.txt`, validated against ttct's deps)
- `registry/models.yaml` (seeded from `result_analysis/output_length_analysis.ipynb` MODEL_NAME_MAP)
- `runner/run.py`
- `runner/test_phase1.sh`
- `scripts/setup_envs.sh`
- Add `registry/environments/.location` to `.gitignore`

**Files to modify:**
- `generate.py` — remove cs4 from `TaskType` enum and `TaskPromptBuilder`
- `main/tasks/__init__.py` — remove cs4 reference
- `tasks/aut_ttcw_cshort/scripts/temp_var_inference*.sh`, `olmo_sft_inference.sh` — strip cs4 lines
- `README.md` — note that cs4 was removed

**Verification steps** (run all of these to confirm Phase 1 is done):

```bash
# 0. Environment setup (first time only)
bash scripts/setup_envs.sh --env modern
# (or with custom prefix)
bash scripts/setup_envs.sh --env modern --prefix /external/storage/conda_envs

# 1. Listing
python runner/run.py --list-tasks         # → aut, ttcw, creative_short, ttct
python runner/run.py --list-models        # → GPT4.1, GPT4.1-mini, Claude3-Sonnet, ...

# 2. Dry-run (no side effects)
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run
python runner/run.py --task ttcw --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run
python runner/run.py --task creative_short --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run
python runner/run.py --task ttct --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run

# 3. Pre-flight check should fail loudly if env is missing
CREATIVITYPRISM_ENV_PREFIX=/tmp/creativityprism-missing \
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label smoke
# → expect: clean error pointing to setup_envs.sh

# 4. Real smoke test (cheap model, few samples)
bash scripts/setup_envs.sh --env modern
python runner/run.py --task aut --model GPT4.1-mini --judge-model GPT4.1-mini --label smoke --limit 5
# Verify: tasks/aut_ttcw_cshort/data/output/smoke/aut/... contains 5 outputs

# 5. Error handling
python runner/run.py --task fake_task --model GPT4.1 --judge-model GPT4.1-mini --label smoke --dry-run
python runner/run.py --task aut --model FakeModel --judge-model GPT4.1-mini --label smoke --dry-run

# 6. cs4 cleanup verification
grep -r "cs4\|CS4" generate.py main/tasks/__init__.py tasks/aut_ttcw_cshort/scripts/ || echo "OK: no cs4 references"

# 7. The wrapper test script runs all of the above
bash runner/test_phase1.sh
```

**Out of scope for Phase 1:**
- Evaluation (deferred to Phase 2)
- Centralized `outputs/` symlinks (deferred to Phase 2)
- SLURM (deferred to Phase 3)
- Other tasks: neocoder, dat, creative_math, creativity_index (deferred to Phase 2)

---

### Phase 2: Evaluation + Centralized Outputs

**Status:** Phase 2A complete (2026-08-01); Phase 2B (evaluation dispatch) and remaining task adapters not started

**Scope:**
- Add evaluation support to existing AUT bundle + TTCT adapters.
- Add adapters + YAMLs for the remaining tasks: **neocoder, dat, creative_math, creativity_index**.
- Implement centralized `outputs/` directory with links. *(Done in 2A.)*

**Adapter changes:**
- Each adapter learns to accept `--inference-only` and `--eval-only` (default = both).
- Each adapter emits `CP_ARTIFACT <kind> <path>` for each phase that actually succeeded. *(Done in 2A for `inference`.)*

**Runner changes:** *(Done in 2A.)*
- Parse artifact markers from adapter stdout while streaming it live to the terminal.
- Create `outputs/{label}/{task}/{canonical_model}/`.
- Link native artifacts into it, with a `.path` reference fallback where symlinks are unavailable.
- Write `metadata.json` (run timestamps, canonical model names, native paths, command used, exit code).

**Phase 2A deviations from the original plan:**

1. The marker is `CP_ARTIFACT <kind> <path>`, not `OUTPUT_PATH=<path>`. The original single-valued marker could not distinguish inference from eval artifacts, and a namespaced prefix is far less likely to collide with arbitrary task stdout. `OUTPUT_PATH=` is still parsed as `inference`.
2. Parsing and materialization live in `runner/artifacts.py`, not inline in `runner/run.py`. The module has zero task-specific knowledge, which is what "the runner is dumb" actually requires, and it is unit-testable without a subprocess.
3. Link names are not always `*.json`. Directory artifacts (the AUT bundle) get an extensionless `{kind}_output` link; file artifacts (TTCT) keep the native suffix, e.g. `inference_output.json`.
4. `metadata.json` is written even when the adapter exits non-zero, so `outputs/` is a complete ledger of attempted runs rather than only successful ones. This is what makes the recorded `exit_code` meaningful.
5. The runner now streams adapter stdout through a pipe instead of `subprocess.call`. Lines are echoed as they arrive, so long GPU runs still show live progress. stderr is left inherited and is never captured.
6. `outputs/` is gitignored.

**Phase 2A verification:**

```bash
bash runner/test_phase2a.sh     # 4 checks, 29 unit tests
bash runner/test_phase1.sh      # 19 checks; must stay green
```

The Phase 2A gate is deterministic: it invokes no models, no paid APIs, and no conda environments.

**Verification steps (Phase 2B, end-to-end):**

```bash
# Run all tasks
python runner/run.py --task all --model GPT4.1-mini --judge-model GPT4.1-mini --label phase2_test --limit 5

# Verify centralized outputs
ls outputs/phase2_test/
# → aut/ ttcw/ creative_short/ ttct/ neocoder/ dat/ creative_math/ creativity_index/

ls outputs/phase2_test/aut/GPT4.1-mini/
# → inference_output (link)  eval_output.json (link)  metadata.json

# Verify links point to real files
readlink outputs/phase2_test/aut/GPT4.1-mini/inference_output
# → ../../../../tasks/aut_ttcw_cshort/data/output/phase2_test/aut/gpt_4.1_mini

# Direct task rerun should still work (links should remain valid)
cd tasks/aut_ttcw_cshort && python run_inference.py configs/aut/inference/gpt.json phase2_test
cat ../../outputs/phase2_test/aut/GPT4.1-mini/inference_output/*.json   # still readable
```

**Out of scope for Phase 2:**
- SLURM (deferred to Phase 3)
- Loader.py (deferred to Phase 3)

---

### Phase 3: SLURM + Analysis Loader

**Status:** SLURM built 2026-08-02 but never submitted to a real cluster; loader complete 2026-08-03

**Scope:**
- Add `--slurm` flag to the runner. — **Done**
- Build `result_analysis/loader.py` for unified output loading. — **Done**

**Runner changes:** *(implemented; see CHANGE_LOG for the design decision)*
- `--slurm` generates an sbatch script wrapping the same command, then calls `sbatch` (unless `--no-submit`).
- SLURM headers come from `runner/slurm_template.sbatch`, with per-task overrides from `registry/tasks/{name}.yaml`, and a third layer of `--slurm-override key=value` flags. An empty value drops the directive.
- Generated scripts go to `slurm_scripts/{label}/{task}_{model}.sbatch`; logs to `slurm_scripts/{label}/logs/`. Both git-ignored.
- The script wraps `runner/run.py`, **not** the adapter, so artifact materialization happens inside the job. Scripts contain no absolute paths.

**`result_analysis/loader.py`:** *(implemented; deviations from this plan noted below)*

```python
import sys; sys.path.insert(0, "result_analysis")
import loader

# Load all outputs for a run
df = loader.load_outputs(run_id="v3")
# → DataFrame: run_id, task, model, sample_id, metric, prompt, output, eval_score

# Filter by task and/or model
df = loader.load_outputs(run_id="v3", task="aut", model="GPT4.1")
```

- One parser function per task (knows the native format).
- Reads from `outputs/{run_id}/...` only, resolving artifacts via `metadata.json` (which records native paths whether or not symlinks were available).
- Reuses canonical model names from `registry/models.yaml`.

**Deviations from the plan, decided during implementation:**

- **A `metric` column was added — 8 columns, not the 7 listed above.** A single `eval_score`
  column would otherwise mix a DAT semantic distance (~86), an n-gram coverage in [0, 1] and
  a binary rubric verdict into one number. `eval_score` is only interpretable next to `metric`.
- **The loader flattens and never aggregates.** One row per scored unit; the notebook takes
  the mean. Deciding what "the score" of a task is belongs in the analysis.
- **pandas is optional.** `load_records()` returns dicts with no third-party import;
  `load_outputs()` imports pandas lazily.
- **A generated-but-unscored unit is kept** as a row with `metric = None`, so the row count
  reflects what ran rather than what the evaluator managed to score.
- **The planned `from result_analysis.loader import load_outputs` works, but only from the
  repo root.** There is no `__init__.py`; it resolves as a namespace package. A notebook in
  `result_analysis/` — which is where `visualization.ipynb` lives — must use `import loader`
  instead. Both spellings are documented in the module docstring.

**Verification steps:**

```bash
# Laptop-runnable gate: script generation, override resolution, fan-out, marker durability
bash runner/test_phase3.sh          # 28/28 as of 2026-08-02

# Loader gate: synthetic fixtures frozen from real artifact shapes, covering the
# scored paths a mock judge cannot produce
bash runner/test_loader.sh          # 20/20 as of 2026-08-03

# SLURM dry-run
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --slurm --no-submit
cat slurm_scripts/v3/aut_GPT4.1.sbatch   # verify sbatch script is well-formed

# Full SLURM submission (real) — NOT YET DONE, no cluster access from the dev machine
python runner/run.py --task all --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm
squeue -u $USER   # verify jobs queued

# Loader test
python -c "
from result_analysis.loader import load_outputs
df = load_outputs(run_id='v3')
print(df.head())
print(df.groupby(['task', 'model']).size())
"
```

**Open on the cluster, cannot be closed locally:** consolidated into
[Read before run](#read-before-run) — `sbatch` accepting the generated directives, whether the
resource defaults suffice, and the `uname`-guarded `cygpath` branches in `_common.sh`.

---

## Security TODOs (must resolve before public release)

1. **Rotate HF token.** ✓ Done (2026-04-08). New token written to `api_keys.json`. Working-tree hardcoded references were removed; token still exists in commit `e0b114f` git history. Run `git filter-repo` before making this repo public if a full history scrub is needed.

2. **Credential file location.** All API keys live in `/api_keys.json` at repo root. This file is gitignored. The runner exports `CREATIVITYPRISM_API_KEYS=<abs path>` so adapters and tasks pick it up without hardcoding paths. New contributors should create their own `api_keys.json` from a template.

---

## Open questions / known unknowns

These are not blockers but should be revisited as the project matures:

1. **Metric formalization.** Currently metrics like `best_avg_rating` are computed in notebooks. Eventually we may want a `registry/metrics/` folder so analysis becomes scriptable. Out of scope for now.

2. **Evaluation cost guardrails.** Eval often calls expensive judge models (e.g. GPT-4.1). Should we add a `--max-cost` or confirmation prompt? Defer until Phase 2 implementation reveals real friction.

3. **Migration of existing analysis notebooks.** The Google Sheet workflow in `visualization.ipynb` should eventually be replaced with `loader.py`-driven analysis. Not blocking; can happen incrementally after Phase 3.

4. **The `main/` folder.** Older incomplete unified framework. **Ignored entirely by this restructuring.** Decide later whether to delete or merge.

---

## Working with this plan in future sessions

When starting a new Claude Code session for any phase:

1. First message should be: *"Read RESTRUCTURING_PLAN.md and implement Phase N."*
2. Claude will load the plan, the memory files (`~/.claude/projects/.../memory/`), and any relevant code.
3. After completing a phase, update the **Status** line in this file from "Not started" to "Complete (date)" and add any deviations from the plan to that phase's section.

Decisions in this document are the result of careful design discussion. Do not re-litigate them without explicit user instruction.