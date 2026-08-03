# CreativityPrism Restructuring — Change Log

A running record of restructuring progress. Companion to [RESTRUCTURING_PLAN.md](RESTRUCTURING_PLAN.md), which is the design source of truth. This file tracks **what has actually been built** and **what is still pending**.

New sessions: read `RESTRUCTURING_PLAN.md` first for design intent, then this file for current state, then verify against the codebase before acting.

---

## Status snapshot (2026-08-01)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Foundation (inference + registry + cleanup) | **Complete + Published to `main_v2` (2026-07-22)** | 4 tasks wired; 19/19 Phase 1 checks pass |
| Phase 2A — Artifact contract + centralized outputs | **Complete (2026-08-01)** | `CP_ARTIFACT` markers, `outputs/` materialization, `metadata.json`; 4/4 gate checks, 29 unit tests |
| Phase 2B — Evaluation dispatch | **Complete (2026-08-01)** | Eval branches wired for all four tasks; 33 unit tests. Not yet run against paid judges. |
| Phase 2C — Remaining task adapters | **Complete (2026-08-01)** | All eight tasks wired; `legacy` env authored; 35 unit tests. Nothing run on the cluster yet. |
| API-only local execution | **Complete (2026-08-02)** | `api` venv env + `CREATIVITYPRISM_FORCE_ENV`; vLLM imports made lazy so API paths need no GPU. |
| Phase 3 — SLURM submission | **Built, not cluster-verified (2026-08-02)** | `--slurm`/`--no-submit`/`--slurm-override`, `runner/slurm.py`, durable artifact markers; 28/28 gate checks. No `sbatch` has ever been run. |
| Phase 3 — Analysis loader | **Complete (2026-08-03)** | `result_analysis/loader.py`; all eight parsers validated against real artifacts; 20/20 gate checks |
| Real-judge verification | **In progress (2026-08-03)** | First runs against paid APIs at `--limit 10`. Surfaced four pre-existing bugs that no mock run could reach — see below. |

---

## Real-judge verification (2026-08-03)

### Why

Every run to date used either a local mock endpoint or no judge at all, so the whole
inference→judge→score path had never been exercised with a model that reasons before it answers.
The mock returns short prose and always parsed to `UNCLEAR`, which looked like a mock limitation.
It was not — it was hiding twelve real bugs.

### What the first paid run found

| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | `ModelWrapper` hardcoded `max_tokens=30` for every provider | The vLLM path honoured the configured `max_new_tokens`; the API path ignored it. **Every `creative_math` API generation was a ~150-character stub**, and judges correctly called the solutions "incomplete" → 0% correctness. | `ModelWrapper` now takes `max_tokens`; `inference_driver.py` passes the configured value. Judges default to `CREATIVITYPRISM_MATH_API_MAX_TOKENS` (512). |
| 2 | Same 30-token cap applied to the judges | Models that explain before answering were cut off before stating a verdict, so `extract_yes_no` saw no verdict. | Same fix. Set the env var to `30` to reproduce old numbers. |
| 3 | Gemini 2.5+ spends the output budget on thinking | With a small budget the response contains **no `parts` at all** — not an error, just nothing. | `thinking_config=ThinkingConfig(thinking_budget=0)` in `_query_gemini`. |
| 4 | Windows cp1252 stdout | A judge writing `✓` raised `UnicodeEncodeError` mid-run, aborting an evaluation after every paid call had already been made. | Runner exports `PYTHONIOENCODING=utf-8` and decodes adapter stdout as UTF-8. |
| 5 | Adapter stdout is a pipe, so Python block-buffers it | `ttct` ran for over an hour printing **nothing at all** — not even its first `print()` — and was mistaken for a hang. | Runner also exports `PYTHONUNBUFFERED=1`. |
| 6 | `infini-gram` lookups fell through to `occurrence = 0` on failure | A failed request is indistinguishable from "n-gram not in corpus". Lost lookups lower coverage, and lower coverage **raises** the reported creativity index, so a throttled run degrades silently and flatteringly. The endpoint 403s about half the time. | 8 retries with exponential backoff and jitter; raises instead of scoring a failed lookup as zero. |
| 7 | Loader joined `creative_math` on `(problem_id, question_number)` | The same problem appears several times with different `k`, so 18 records collapsed to **10 keys** and collisions inherited the last record's verdicts. The loader reported correctness `27.78%` where the evaluation script printed `55.56%`. | Join key now includes `k`; loader and script agree exactly. |
| 8 | Loader emitted `ttct` placeholder rows | Rows outside the configured subset carry the literal output `SKIPPED` and are not marked `skip`. A 10-item run produced 630 rows, 600 of them placeholders. | Loader drops variants whose output is the `SKIPPED` sentinel; the same run now yields 30 rows. |
| 9 | `neocoder`'s timeout helper started **non-daemon** threads | A timed-out thread is abandoned, never killed. The interpreter's shutdown handler joins non-daemon threads, so the process hung **forever after its results had already been written**. Observed: correctness finished and saved at 09:25, the process was still alive and idle 20 minutes later. On a scheduler this burns the entire wall-clock allocation. | `thread.daemon = True` in `function_with_timeout`. No score changes — the `TimeoutError` path is untouched and the abandoned thread's result was already discarded. |
| 10 | Adapters inherited the runner's **stdin** | Tasks execute model-generated code, and Codeforces solutions read stdin. Attached to a terminal, that code blocks on a prompt nobody is watching. `neocoder` scored **55 of 60** generations as `code execution timeout` for this reason alone; with stdin closed the same artifacts give **0 of 60** timeouts. | `stdin=subprocess.DEVNULL` in `run_adapter`, which is what a batch scheduler would have supplied anyway. |
| 11 | A diagnostic `print` could kill an evaluation | The `infini-gram` retry handler prints from eight worker threads to one redirected stdout. On Windows that raised `OSError: [Errno 22]` *from inside the exception handler*, ending a 40-minute evaluation. | The print is wrapped so it can never propagate. |
| 12 | `extract_yes_no()` matched substrings | `"NO"` is inside `NOT`, `CANNOT`, `KNOW`, `NOTE` and `NOVEL`, and `"YES"` was tested first and anywhere in the string, so `"The answer is not YES"` scored `YES`. A judge opening with "I need to know whether…" was recorded as a `NO` vote — observed live. | All three copies now take the first **whole-word** `YES`/`NO`, case-insensitively, since every prompt asks for the verdict before the explanation. Each copy keeps its original fallback, and a reply with no verdict token becomes `UNCLEAR`, which every downstream tally already counts exactly like `NO`. |

Retired judge model IDs were also replaced: `claude-3-7-sonnet-20250219` →
`claude-sonnet-4-5-20250929`, `gemini-2.0-flash` → `gemini-2.5-flash`. Both must be changed in
**two** places — `configs/eval_creative_math.json` *and* the hardcoded `JUDGE_MODELS` dict.

### Deliberately not fixed

`ttct` scores only the `cot` variant: `ttct_evaluation.py` hardcodes `SKIPPED` for `basic` and
`instructive` in both branches under a `# TODO: change this`. Upstream behaviour, left alone.
Separately, ttct contributes **no numeric score** to the loader today, even though the judge
emitted a clean `### Scores ###` block with four named dimensions on 10 of 10 samples. Parsing it
would make ttct loadable without touching any existing metric, but deciding what the benchmark
reports is the maintainer's call.

### Also in this slice

`neocoder` gained `--limit`. It previously had no way to process fewer than all 199 problems, so
the smallest possible run was 199 × 6 rounds = **1194 paid calls**. The limit caps problems, not
prompts, so each problem keeps its full denial ladder.

### Coverage of the `real10` verification run

Seven of eight tasks completed end to end against real paid APIs at n≈10:
`aut`, `dat`, `ttcw`, `creative_short`, `creative_math`, `ttct`, `neocoder`.
`creativity_index` completed inference but **could not complete evaluation**, because the public
`infini-gram` endpoint refused roughly half of all requests from this network. That is an external
service limit, not a repository defect.

All eight tasks are now visible to the loader (764 rows), including `creativity_index` and
`neocoder`. `creativity_index` contributes inference rows with no score, for the reason above.

One unresolved observation, deliberately **not** acted on: even after the stdin fix removed every
spurious timeout, `neocoder` correctness is `0/60`. The generated code is executable and all 60
samples do define `solve()`, but the per-test-case path records `None` for every case, i.e.
`solve()` raises each time. That is the upstream harness (`parse_code` truncation, `mock_input`
line arity, and the hardcoded `solve()` entry point), not the runner, and changing it would move
published numbers. Two smaller oddities from the same artifacts: `new_techniques_ratio` is `1.0`
for all 55 scored rows, and 5 of 60 generations are absent from the creativity CSV because the
upstream evaluator drops them. See `RESTRUCTURING_PLAN.md` → "Read before run".

---

## Commit ledger

Anchors each phase to the commit that implemented it, so a future session can jump straight to the diff. This table deliberately does **not** duplicate file lists or line numbers — `git show <hash> --stat` and `git show <hash> -- <path>` are the authoritative per-file record and never go stale. What lives here instead is the mapping and the verification state, which git does not capture.

| Commit | Date | Phase | Scope | Gates at commit time | Pushed to `personal/main_v2` |
|--------|------|-------|-------|----------------------|------------------------------|
| `d6e4cf4` | 2026-08-03 | 3 (verification) | Eleven bugs found by the first real-judge run of all eight tasks: API token caps, Gemini thinking budget, runner stdout encoding / buffering / stdin, `infini-gram` silent zeros, `neocoder` non-daemon threads, a fatal diagnostic print, and two loader join bugs; `neocoder --limit` | Phase 1 19/19, 2A 4/4, Phase 3 28/28, API env 21/21, loader 20/20 | Pending |
| `5ddbe04` | 2026-08-03 | 3 (loader) | `result_analysis/loader.py` and `runner/test_loader.sh`; eight per-task parsers validated against real artifacts | Phase 1 19/19, 2A 4/4, Phase 3 28/28, API env 21/21, loader 20/20 | Pending |
| `b00ed40` | 2026-08-02 | API-only + 3 (SLURM) | `api` venv environment and lazy vLLM imports; `--slurm`/`--no-submit`/`--slurm-override`, `runner/slurm.py`, `runner/slurm_template.sbatch`; durable `.cp_artifacts` sidecar; `.gitattributes` LF pin for `*.sh`; pre-existing bugs fixed in `creative_math` (uninvoked cleaning step, hardcoded model), `neocoder` and `ttct` | Phase 1 19/19, 2A 4/4, Phase 3 28/28, API env 21/21 | Pending |
| `0d35eef` | 2026-08-01 | 2C batch 2 | Task schema applied via a per-file `yaml-language-server` modeline, since the workspace `yaml.schemas` setting only takes effect after an extension reload | Phase 1 19/19, Phase 2A 4/4 | Pending |
| `e7d852f` | 2026-08-01 | 2C batch 2 | `dat` registry entry and adapter; `evaluate_dat.py` GloVe/word-list paths and `--output-path`; `.gitignore` gap for `tasks/math_n_index/data/`; task JSON schema | Phase 1 19/19, Phase 2A 4/4 (35 unit tests) | Pending |
| `7fd6da3` | 2026-08-01 | 2C batch 1 | `neocoder`, `creative_math`, `creativity_index` registry entries and adapters; `legacy` env; provider-key export; removal of the `data[:100]` cap in the creativity-index driver | Phase 1 19/19, Phase 2A 4/4 (35 unit tests) | Yes |
| `747a5ba` | 2026-08-01 | 2A + 2B | Artifact contract (`CP_ARTIFACT` markers, `runner/artifacts.py`, `outputs/` materialization, `metadata.json`) and evaluation dispatch for all four wired tasks; credential path resolution in the bundled task driver | Phase 1 19/19, Phase 2A 4/4 (33 unit tests) | Yes |
| `56cfbd3` | 2026-07-22 | 1 | Registry-driven runner: `registry/{tasks,adapters,models.yaml}`, `runner/run.py`, Phase 1 gate; 4 tasks wired for inference | Phase 1 19/19 | Yes |
| `4705a83` | 2026-05-14 | — | Pre-restructuring baseline; also the tip of public `main` | n/a | n/a (public `main`) |

To see exactly what a commit changed:

```bash
git show 747a5ba --stat                       # files touched
git show 747a5ba -- registry/adapters/aut.sh   # one file's hunks
git log -p --follow runner/artifacts.py        # one file's whole history
```

Rationale for each change lives in the per-phase sections below; the "Deviations from the plan" tables are the part that git cannot tell you.

---

## Workspace and publication checkpoint (verified 2026-08-01)

- Canonical development worktree: `creativityprism_v2-mainv2-clean`.
- Local branch `main_v2_publish` tracks remote branch `personal/main_v2`.
- Published Phase 1 commit: `56cfbd3ce564535a2416cb847641660da2a70118`.
- Published Phase 2A + 2B commit: `747a5ba1dcc4848410d807f921991bd93d122fb9`.
- Commit author and committer: `Joey Hou (MS) <joeyhou.work@gmail.com>`.
- Local HEAD, upstream, and GitHub remote hashes were independently verified equal after push; ahead/behind was `0/0`.
- Public `main` remained at `4705a830501e47b999481a0ec0c62ac2cca10c86` during publication.
- Pushes are performed from the VS Code Git UI, not from an agent-run terminal — see `RESTRUCTURING_PLAN.md` "Publishing procedure" for why.
- Original `creativityprism_v2/` worktree remains an intentionally untouched, dirty, read-only migration source. It must not be used for new development or deleted before asset review.
- Next slice: build the `legacy` env on the cluster and run Phase 2 end-to-end, then Phase 3.

See the top of `RESTRUCTURING_PLAN.md` for the new-session startup gate and safety constraints.

---

## Phase 3 — SLURM submission — Built, not cluster-verified (2026-08-02)

### Why

Phase 2 left the runner able to execute a task locally but not to queue one. The blocker
recorded in the plan was that `CP_ARTIFACT` stdout markers cannot survive a batch job: the
adapter runs detached and its stdout goes to a SLURM log the runner never reads.

### The design decision worth reviewing

The plan offered two ways out — parse the job's stdout log after it finishes, or make the
markers durable. Neither was chosen as specified. What is implemented instead:

**The batch job re-invokes `runner/run.py`, not the adapter.** The generated script runs
the same command line minus the SLURM flags, so the runner *inside the job* does the
artifact linking and writes `metadata.json`. A cluster run and a laptop run therefore
produce byte-identical `outputs/` trees, submission stays asynchronous, and there is no
collect step to forget. The alternative — submitting the adapter directly and reconciling
afterwards — would have meant two code paths for the same contract.

**Markers were made durable anyway, as a second line of defence.** `emit_artifact` now also
appends `CP_ARTIFACT <kind> <path>` to `$OUTPUT_DIR/.cp_artifacts`. If a job is killed
between the adapter finishing and the runner materializing, the record survives.
`materialize_run_outputs` merges the sidecar with stdout and **prefers stdout**, because a
stale sidecar can persist in a `(label, task, model)` directory that is re-run.

### What was built

| Piece | Notes |
|---|---|
| `--slurm`, `--no-submit`, `--slurm-override key=value` | `--task all` fans out to one script and one queue slot per task |
| `runner/slurm.py` | Directive resolution, script rendering, `sbatch` invocation |
| `runner/slurm_template.sbatch` | Defaults only; `#~` lines are template documentation and are stripped on render |
| `slurm` block in `registry/tasks/_task.schema.json` | Per-task directive overrides; `cpus_per_task` accepted for `cpus-per-task` |
| `runner/test_phase3.sh` | 28 checks, all laptop-runnable |

Directives resolve template → task YAML → `--slurm-override`, and an empty value drops the
directive. That is how an API-model run stops queueing for a GPU it will never use:
`--slurm-override gres=`.

Generated scripts contain **no absolute paths**. The repo root comes from the script's own
location and the interpreter is `python3` from `PATH`, so a script generated on Windows still
runs on a Linux cluster. This is checked by the gate.

### Deliberately not chosen

Per-task `slurm:` blocks are shipped **commented out**. Partition names, accounts and time
limits are properties of a specific cluster, and a plausible-looking wrong default is worse
than an obvious missing one. The one substantive note left in a task file is on `dat`, where
eval parses a 5.3 GB GloVe file and may need more memory than the template default.

### Verification

28/28 Phase 3 checks; Phase 1 19/19 and Phase 2A 4/4 still pass after the `artifacts.py` and
`_common.sh` changes.

### Caveat carried forward

**No `sbatch` has ever been run.** The gate can prove the script is well-formed, portable and
carries the right flags; it cannot prove the partition exists, that the directives are
accepted, or that the resources are sufficient. Submit one `--limit` job before a full matrix.

---

## API-only local execution — Complete (2026-08-02)

### Why

The cluster is unavailable for roughly half a month. This slice makes every task runnable
on a laptop against hosted APIs, with no conda, no GPU and no vLLM.

### What was built

- `registry/environments/api.requirements.txt` — a third environment spec. `scripts/setup_envs.sh`
  now dispatches on the spec **file extension**: `.yml` and `.txt` stay conda, `.requirements.txt`
  is a plain `venv`. No vLLM, CPU-only torch.
- `CREATIVITYPRISM_FORCE_ENV` — overrides the env an adapter activates, so a task whose
  registry entry says `modern` can run in `api`. `runner/artifacts.py` records the effective
  env in `metadata.json`.
- `runner/run.py check_env_model_compat` — exits 5 with a readable message when an open-weights
  model is requested in the `api` env, instead of failing deep inside vLLM. Runs during `--dry-run`.
- vLLM imports made lazy in 7 modules, so importing an inference driver no longer requires a GPU.
- `registry/adapters/_common.sh` — `cygpath` conversion for env activation and artifact paths,
  plus `add_pythonpath()`. Only triggers under MINGW/MSYS/CYGWIN; the cluster path is unchanged.
- `runner/test_api_env.sh` (21 checks) and `api_keys.example.json`.

### Pre-existing bugs found by being the first to run the API path end-to-end

| Bug | Scope | Fix |
| --- | --- | --- |
| `dp_generator.py` wrote JSONL on the API branch while all three readers used `json.load` | neocoder, API models only — the vLLM branch wrote a real array | Restored the array write, and seeded `records` from the existing file so `enumerate_resume` no longer drops previously completed batches when the array is rewritten |
| `calculate_creativity` asserts both `correctness` and `techniques`, but the adapter ran correctness before detection, leaving the two fields in different files | neocoder, all models — the creativity step could never have run | Reordered to detection → correctness → creativity and pointed creativity at the correctness output |
| `--limit N` truncated ttct inference, but eval asserts one row per `basefile.csv` row | ttct, all models | `--limit` now keeps all 700 rows and queries the first N of each scored question type, marking the surplus `skip` so the judge skips them too |
| `creative_short_story.py` imported `story_metrics` at module scope, so a missing spacy broke aut and ttcw eval as well | aut, ttcw, creative_short | Import moved into the two methods that use it |
| Six missing dependencies (`scikit-learn`, `unidecode`, `sacremoses`, `openpyxl`, `scipy`, `gensim`) | creativity_index, dat, creative_short | Added to the api spec; found by AST-scanning task imports rather than by repeated re-runs |
| Eval reads `sample["cleaned_response"]`, a field only `clean_data_creative_math.py` produces — and the adapter never called it | creative_math, all models — eval always died with `KeyError` | The adapter now runs the cleaning step at the start of eval, skipping it when the field is already present. The script gained an `openai` backend (default stays vllm + Llama-3.3-70B) and records `cleaner_model` per item |

### Verification

Inference runs for all 8 tasks; evaluation runs for aut, ttcw, ttct, creative_short,
neocoder and creative_math against a local mock OpenAI server. Gates: Phase 1 19/19,
Phase 2A 4/4, api env 21/21.

Two tasks still cannot be evaluated locally, for reasons that are **not** regressions:

- `dat` needs GloVe 840B.300d (~2GB), not shipped. The task already prints a clean message.
- `creativity_index` needs the gated HF repo `meta-llama/Llama-2-7b-hf`.

### Comparability caveat for `creative_math`

The cleaner is fixed and independent of the model under test, because a cleaner that tracked
the evaluated model would make api-model and open-model scores non-comparable. It defaults to
`vllm`/Llama-3.3-70B everywhere, so the cluster path is unchanged; the api env exits 4 with
instructions rather than silently substituting a smaller cleaner. Opt in with
`CREATIVITYPRISM_MATH_CLEANER_BACKEND=openai` (and optionally
`CREATIVITYPRISM_MATH_CLEANER_MODEL`), using the same value across the whole comparison.
Each item records `cleaner_model`.

### Line endings

Added `.gitattributes` pinning `*.sh` and the env specs to LF. The repo has
`core.autocrlf=true` and no prior attributes file, so any future checkout would have rewritten
the adapters to CRLF and broken them under Git Bash and on the cluster.

### Caveat carried forward

The `cygpath` work in `_common.sh` also touches the cluster code path. It is guarded on
`$OSTYPE`, but it has not been exercised on the cluster.

---

## Phase 2C batch 2 — Complete (2026-08-01)

Wires the last task, `dat`. All eight benchmark tasks now run through `runner/run.py`.

### What was built

| File | Change |
|------|--------|
| `registry/tasks/dat.yaml`, `registry/adapters/dat.sh` | **New.** |
| `tasks/neocoder_dat/steps/evaluate_dat.py` | The GloVe and word-list paths were hardcoded to a site-local AFS location. They now come from `--glove-path` / `--words-path`, then `$CREATIVITYPRISM_GLOVE_PATH` / `$CREATIVITYPRISM_DAT_WORDS`, then `embeddings/glove/glove.840B.300d.txt` and the bundled `words.txt`. A missing GloVe file now fails with download instructions instead of a bare `FileNotFoundError`. Added `--output-path`. |
| `tasks/neocoder_dat/.gitignore` | Added `embeddings/`. |
| `tasks/neocoder_dat/README.md` | Documented the GloVe download and the `--output-path` caveat. |
| `.gitignore` | Added `tasks/math_n_index/data/{outputs,evaluations}/`. |
| `runner/test_phase2a_artifacts.py` | Added `dat` to `ALL_ADAPTERS`. |
| `.vscode/settings.json`, `registry/tasks/_task.schema.json` | **New.** |

### Deviations from the plan

| # | Deviation | Why |
|---|-----------|-----|
| 1 | `dat` sets `limit_supported: true`, with `--limit` mapped to `--repeat` | DAT has one fixed prompt rather than a dataset, so the repeat count *is* the sample count. This is the same semantic `--limit` has elsewhere, not a reinterpretation. |
| 2 | `evaluate_dat.py` gained `--output-path` and an overwrite guard | Its only output rule was `result_path.replace("inference", "evaluation")`, which **overwrites the inference file** when the path contains no `inference` segment. That is silent data loss, and the file was already being edited for the GloVe path. |
| 3 | The adapter globs the inference directory instead of reconstructing the filename | The name embeds `int(temperature * 100)`, which floating point makes unreliable to reproduce in shell (`0.29` truncates to `28`). The directory is run-scoped, so exactly one file is present; the adapter fails loudly if that is not true. |
| 4 | `.gitignore` gained explicit `tasks/math_n_index/data/` entries | The pre-existing `*/data/outputs/*` pattern only matches **one** path segment before `data/`, so it never covered `tasks/math_n_index/...`. Batch 1's adapters would have left generated outputs untracked-but-visible. Caught while adding the GloVe ignore. |
| 5 | A JSON schema and workspace setting were added for `registry/tasks/*.yaml` | schemastore maps `**/tasks/*.yaml` to the Ansible tasks schema, so every field in every task YAML showed as an editor error. Purely cosmetic, but it also now validates the registry contract in-editor. |

### Verification

- `bash runner/test_phase1.sh` — 19 passed, 0 failed. `bash runner/test_phase2a.sh` — 4 passed, 0 failed (35 unit tests).
- `python runner/run.py --list-tasks` shows all 8 tasks; `--task dat --limit 5 --dry-run` exits 0.
- `evaluate_dat.py` guards exercised with the scorer stubbed: the missing-GloVe path raises the documented message, `--output-path` equal to the input is refused, and a normal run creates parent directories, writes scores, and leaves the input untouched.
- `git check-ignore` confirms `tasks/neocoder_dat/embeddings/`, `tasks/math_n_index/data/outputs/` and `.../data/evaluations/` are now ignored.
- **Not verified:** nothing has been run on the cluster. No `legacy` env built, no GloVe downloaded, no model loaded, no paid judge called.

---

## Phase 2C batch 1 — Complete (2026-08-01)

Wires three of the four remaining tasks: `neocoder`, `creative_math`, `creativity_index`.
`dat` is deliberately held back to batch 2 because it needs a GloVe download convention
that does not exist yet.

### What was built

| File | Change |
|------|--------|
| `registry/environments/legacy.yml` | **New.** vllm 0.5.3.post1 / torch 2.3.1 / Python 3.11 / CUDA 12.1 for the `neocoder_dat` bundle. `scripts/setup_envs.sh` auto-discovers `registry/environments/*.yml`, so no script change was needed. |
| `registry/tasks/{neocoder,creative_math,creativity_index}.yaml` | **New.** |
| `registry/adapters/{neocoder,creative_math,creativity_index}.sh` | **New.** |
| `registry/adapters/_provider_keys.py` | **New.** Reads `api_keys.json` and prints shell-quoted `export` lines for the provider env vars the two bundles read natively. |
| `registry/adapters/_common.sh` | Added `export_provider_keys`. Also fixed the `nvjitlink` `LD_LIBRARY_PATH` probe, which hardcoded `python3.12` and would have silently skipped the Python 3.11 `legacy` env. |
| `registry/models.yaml` | Added `neocoder_dat` aliases (13 models) and `math_n_index` aliases (18 models). A missing alias is intentional and means the bundle does not support that model. |
| `tasks/math_n_index/src/inference/creative_index_inference.py` | Removed the unconditional `data[:100]` cap; added `test_size`. |
| `tasks/math_n_index/src/inference/creative_math_inference.py` | Added `test_size`. |
| `tasks/math_n_index/src/inference/inference_driver.py` | Closed-source API keys now resolve from provider env vars, falling back to the config value. |
| `tasks/math_n_index/src/evaluation/creative_math_eval_api.py` | Config path now comes from `CREATIVITYPRISM_MATH_EVAL_CONFIG`; judge keys come from provider env vars instead of placeholders. |
| `tasks/neocoder_dat/src/models/model.py` | `CACHE_DIR` now honours `HF_HOME` (exported by `_common.sh`) instead of a hardcoded `/scratch365/...` path. |
| `runner/test_phase2a_artifacts.py` | +2 tests (35 total). Split the `$NATIVE_OUT` assertion away from the guard-ordering assertions so directory-artifact adapters are covered, and added a test that every `registry/adapters/*.sh` is listed in the gate. |

### Deviations from the plan

| # | Deviation | Why |
|---|-----------|-----|
| 1 | `legacy.yml` only; no `legacy.txt` | `.txt` files are `conda list --export` snapshots. One cannot be authored for an environment that has never been built. |
| 2 | `--limit` is expressed as an exact `test_size`, not as a float `portion` | The tasks only exposed `portion`, and converting an integer limit to a fraction is lossy and off-by-one prone. `test_size` matches the convention the AUT bundle already uses and that `ExactLimitTests` covers. |
| 3 | Provider keys travel through env vars, not through the generated configs | `tasks/neocoder_dat` already reads `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GENAI_API_KEY`/`DEEPSEEK_API_KEY` natively, so this needed zero patching there. It also keeps secrets out of any file the adapter writes. `api_keys.json` stays model-keyed for the older tasks; the provider block is additive and optional. |
| 4 | `neocoder` and `creativity_index` announce a **directory** artifact | NeoCoder's output filename embeds a sample count computed inside the task; `creativity_index` writes one file per domain. Both directories are run-scoped, so the artifact is still unambiguous. |
| 5 | `--judge-model` is a no-op for all three tasks | `creative_math` uses a fixed three-judge majority vote, `creativity_index` uses an n-gram metric with no judge, and NeoCoder's technique detector hardcodes `gpt-4-turbo`. Honouring the flag would silently produce numbers that are not comparable to the published ones. |
| 6 | One registry task covers all three `creativity_index` domains | The benchmark reports Creativity Index over book/poem/speech together. The adapter loops; `CREATIVITYPRISM_INDEX_DOMAINS` narrows it. |
| 7 | `creativity_index` eval defaults to the full 5..12 `min_ngram` sweep | That is what the bundled `book_creative_par.sh` and friends do. `CREATIVITYPRISM_INDEX_MIN_NGRAM` allows a cheaper single-`n` run. |
| 8 | `creative_math_eval_api.py` gained an env-var config override | It loads its config at **import time** from a fixed relative path, so an adapter cannot pass one as an argument. The checked-in `configs/eval_creative_math.json` is also the wrong shape for it (`experiments_list` vs. flat) and would raise `KeyError`. |

### Verification

- `bash runner/test_phase1.sh` — 19 passed, 0 failed.
- `bash runner/test_phase2a.sh` — 4 passed, 0 failed (35 unit tests).
- `bash -n` clean on all eight adapters; `py_compile` clean on every patched Python file; every registry YAML parses.
- `python runner/run.py --list-tasks` shows 7 tasks; `--dry-run` exits 0 for all three new tasks.
- `--limit 5` on `neocoder` is rejected with exit 5, matching `limit_supported: false`.
- `_provider_keys.py` exercised against a synthetic file: placeholders and empty strings skipped, values with spaces quoted, preexisting env vars left untouched, missing file is a silent no-op.
- **Not verified:** nothing has been run on the cluster. No `legacy` env has been built, no model loaded, no paid judge called. The first real run is expected to surface issues no static gate can catch.

---

## Phase 2A — Complete (2026-08-01)

### What was built

| File | Change |
|------|--------|
| `runner/artifacts.py` | **New.** Marker parsing, link/reference materialization, `metadata.json` writer. No task-specific logic. |
| `runner/run.py` | Streams adapter stdout via `Popen` instead of `subprocess.call`, captures it, and materializes `outputs/{label}/{task}/{model}/` after every invocation. `--output-dir` now comes from `artifacts.centralized_output_dir`, so the path handed to adapters and the path the runner materializes cannot drift. |
| `registry/adapters/_common.sh` | Added `emit_artifact <kind> <path>`; rejects kinds other than `inference`/`eval`. |
| `registry/adapters/{aut,ttcw,creative_short,ttct}.sh` | Replaced the unconditional trailing `echo "OUTPUT_PATH=..."` with `emit_artifact inference "$NATIVE_OUT"` **inside** the inference guard. |
| `runner/test_phase2a_artifacts.py` | **New.** 29 tests: marker parsing, linking, path isolation, metadata, missing artifacts, reruns, symlink targets, adapter source contract, stdout streaming. |
| `runner/test_phase2a.sh` | **New.** Phase 2A gate: unit tests, dry-run has no side effects, adapters receive the centralized `--output-dir`, `outputs/` is gitignored. |
| `.gitignore` | Added `/outputs/`. |

### The contract

```text
CP_ARTIFACT inference <abs path>
CP_ARTIFACT eval <abs path>
```

Emitted only for a phase that actually ran and succeeded. `OUTPUT_PATH=<path>` is still parsed as `inference` for backward compatibility; an explicit marker overrides it.

### Deviations from the plan

| # | Deviation | Why |
|---|-----------|-----|
| 1 | Marker is `CP_ARTIFACT <kind> <path>`, not `OUTPUT_PATH=<path>` | The old marker cannot express inference vs. eval, and a namespaced prefix will not collide with arbitrary task stdout. Legacy form still accepted. |
| 2 | Logic lives in `runner/artifacts.py`, not inline in `run.py` | Unit-testable without subprocesses; keeps `run.py` readable. Still zero task-specific knowledge. |
| 3 | Link names are not always `*.json` | Directory artifacts (AUT bundle) get `inference_output`; file artifacts (TTCT) keep the native suffix. |
| 4 | `.path` reference file when symlinks are unavailable | The plan permits "link to **or reference**". Windows without Developer Mode cannot create symlinks; the run must not fail over a convenience layer. Verified live on this machine (`WinError 1314` → clean fallback). |
| 5 | `metadata.json` is written even when the adapter fails | Makes the recorded `exit_code` meaningful and `outputs/` a complete ledger. |
| 6 | Contract violations warn, never fail | A nonexistent announced path is recorded as `"exists": false` and warned on stderr. Adapter exit codes stay the only failure signal, so a path-shape mismatch cannot break a real GPU run. |

### Verification

- `bash runner/test_phase2a.sh` — 4 passed, 0 failed (29 unit tests).
- `bash runner/test_phase1.sh` — 19 passed, 0 failed (unchanged baseline).
- End-to-end: a throwaway probe task exercised the real `python runner/run.py` path and produced `outputs/e2e_probe/_e2e_probe/GPT4.1/{inference_output.path,metadata.json}` with the correct native path, command, and exit code. The probe task, adapter, and outputs were removed afterwards.
- No models, paid APIs, or conda environments are touched by either gate.

---

## Phase 2B — Complete (2026-08-01)

### What was built

| File | Change |
|------|--------|
| `registry/adapters/{aut,ttcw,creative_short}.sh` | Added an eval branch that writes a second ephemeral config (same `run_id` as inference, `task` = the evaluator's dispatch string, `model_name` = the judge), runs `run_evaluation.py`, then emits `CP_ARTIFACT eval <run dir>/eval_output_cleaned.json`. |
| `registry/adapters/ttct.sh` | Added an eval branch calling `src/evaluation/ttct_evaluation.py` with `-infer_model_name`, `-eval_model_name`, `-run_id`, and `-api_key_path`, then emitting `CP_ARTIFACT eval data/evaluations/<run_id>/<model_short>.json`. |
| `tasks/aut_ttcw_cshort/src/driver.py` | Credentials now resolve from `CREATIVITYPRISM_API_KEYS` when it names an existing file, falling back to the historical `./api_keys.json`. |
| `runner/test_phase2a_artifacts.py` | +4 tests (33 total): eval marker sits behind an eval guard and after the inference branch; bundled eval reuses the inference `run_id`; TTCT eval is pinned to the inference run; adapters never copy the credentials file. |

No change was needed in `runner/run.py` or `runner/artifacts.py` — the Phase 2A contract already carried the `eval` kind.

### Deviations from the plan

| # | Deviation | Why |
|---|-----------|-----|
| 1 | The eval artifact is a **file** (`eval_output_cleaned.json`), not the run directory | The bundled evaluator writes back into the inference directory, so announcing the directory would make the `eval` link an exact duplicate of the `inference` link. The per-item file is also what the analysis notebooks read most. |
| 2 | `driver.py` credential resolution was patched | The task read a task-relative `./api_keys.json` while the runner exported a repo-root path, so an API judge could never find keys. Mirrors the Phase 1 `ttct_inference.py` precedent. The fallback keeps existing cluster setups working. |
| 3 | Credentials are **not** copied into the task directory | Duplicating a secrets file into the repo tree is a security regression; a gate test now forbids `cp`/`mv`/`ln` of `CREATIVITYPRISM_API_KEYS` in adapters. |
| 4 | `creative_short` still requires `--judge-model` | Its evaluation is fully automated and ignores the judge. Relaxing the runner's mandatory-judge rule per task was deferred rather than special-cased. |
| 5 | The six ephemeral-config heredocs were left duplicated | Factoring them into `_common.sh` would break the Phase 1 assertion that each adapter's source literally contains its own task-qualified `run_id`. Kept in the cleanup backlog so a refactor diff never mixes with a behavior diff. |
| 6 | TTCT `-temp`, `-summary`, `-demo`, `-pairwise` are not forwarded | `-temp` only selects a fallback input dir when `-run_id` is empty; the other three are `type=bool`, where argparse treats any non-empty string as `True`, so forwarding them cannot express `false`. |

### Verification

- `bash runner/test_phase2a.sh` — 4 passed, 0 failed (33 unit tests).
- `bash runner/test_phase1.sh` — 19 passed, 0 failed (unchanged baseline).
- `bash -n` clean on all five adapter scripts.
- Materialization probe: an `inference` marker (directory) and an `eval` marker (file) produce two distinct entries in `metadata.json` and two distinct links.
- **Not verified:** no evaluation has been run against a real judge model. The first cluster run is expected to surface runtime issues that no static gate can catch.

---

## Phase 1 — Complete (2026-04-12)

### What exists in the codebase

- **Registry**
  - `registry/tasks/`: `aut.yaml`, `ttcw.yaml`, `creative_short.yaml`, `ttct.yaml`
  - `registry/adapters/`: `aut.sh`, `ttcw.sh`, `creative_short.sh`, `ttct.sh`, plus shared `_common.sh`
  - `registry/environments/`: `modern.txt`, `modern.yml`, `cluster_env.sh.example`
  - `registry/models.yaml` (seeded from `result_analysis/output_length_analysis.ipynb` MODEL_NAME_MAP)
- **Runner**
  - `runner/run.py` — supports config/CLI submission, mandatory judge model, labels, positive sample limits, dry runs, and inference/eval mode selection
  - `runner/test_phase1.sh` — structural, validation, forwarding, and behavior gate
  - `runner/test_phase1_behavior.py` — dependency-light tests against the real task prompt producers
- **Scripts**
  - `scripts/setup_envs.sh` — idempotent conda env creator with `--prefix`, `--env`, `--force`
- **Smoke-test artifacts**
  - `tasks/aut_ttcw_cshort/data/output/smoke_phase1/` (proves end-to-end runs landed in native paths)

### Deviations from the plan (already in RESTRUCTURING_PLAN.md, repeated for visibility)

1. `tasks/ttct/src/inference/ttct_inference.py` patched to accept `-run_id` to align output paths across the AUT bundle and TTCT.
2. Adapters share `registry/adapters/_common.sh` for arg parsing, conda activation, and `models.yaml` lookup.
3. Adapters generate ephemeral one-shot JSON configs instead of reusing the legacy per-model files under `tasks/aut_ttcw_cshort/configs/`. `registry/models.yaml` is the sole alias source.
4. `registry/environments/.location` pre-created with `/ix1/xli/joh227/conda_envs` (gitignored).

### Post-completion fixes (2026-04-12)

| # | Bug | Fix location |
|---|-----|--------------|
| 1 | `--limit` not wired through adapters | `aut.sh`/`ttcw.sh`/`creative_short.sh` inject `test_size` into ephemeral config; `ttct_inference.py` gained `-num_samples`; `ttct.sh` passes it |
| 2 | `PYTHONPATH: unbound variable` under `set -u` | `ttct.sh` uses `${PYTHONPATH:-}` |
| 3 | `transformers 5.5.3` incompatible with vLLM 0.7.2 | Pinned `transformers>=4.49.0,<5.0.0` in `registry/environments/modern.yml` |
| 4 | Triton/vLLM caches write into quota-limited `/ihome` | `_common.sh::activate_env` exports `TRITON_CACHE_DIR`, `TORCH_HOME`, `VLLM_NO_USAGE_STATS=1`, plus configurable `HF_HOME` |

### Behavior hardening (2026-07-22)

| # | Bug or gap | Fix location |
|---|------------|--------------|
| 5 | AUT, TTCW, and Creative Short shared `data/output/{run_id}/{alias}` and could overwrite one another | Their adapters now use `data/output/{run_id}/{task}/{alias}` for both logical run IDs and the announced artifact path |
| 6 | The three bundled prompt producers returned `N+1` items for `--limit N` | Their post-append stop checks now use `>=`; real producer regressions cover limits 1, 3, above-dataset, and internal unlimited mode |
| 7 | Public `limit <= 0` semantics were undefined, and all task YAMLs still claimed no limit support | `runner/run.py` accepts only positive limits, validates task capability, and all four verified YAMLs set `limit_supported: true` |
| 8 | Structural smoke tests could not catch sample-count or path-isolation regressions | Added `runner/test_phase1_behavior.py` and integrated it into `runner/test_phase1.sh` |

### Smoke test results — run-id `smoke_phase1`

| Task | Model | Type | Items | Result |
|------|-------|------|-------|--------|
| aut | GPT4.1-mini | API | 63 (full) | PASS |
| ttcw | GPT4.1-mini | API | 4 (requested limit 3) | Exposed off-by-one; fixed 2026-07-22 |
| creative_short | GPT4.1-mini | API | 4 (requested limit 3) | Exposed off-by-one; fixed 2026-07-22 |
| ttct | GPT4.1-mini | API | 3×3 formats (limit 3) | PASS |
| aut | Qwen2.5-7B | vLLM (4×L40S) | 4 items × 5 rounds | PASS |

These are the historical API/GPU runs; they were not rerun on 2026-07-22. The current deterministic Phase 1 gate passes **19/19** checks and executes the real sample-selection code for all four tasks without model or API calls. It verifies exact positive limits, aligned TTCT prompt formats, public rejection of non-positive limits, CLI/config forwarding, and task-qualified native paths.

---

## Phase 2 — In progress

### Confirmed gaps (verified 2026-08-01)

- **No environment has been built for `legacy`.** `registry/environments/legacy.yml` is authored but never installed, so its pins are unproven.
- **No GloVe vectors are present**, so `dat` evaluation cannot run until they are downloaded.
- **No judge has actually been run.** Eval dispatch is wired and statically verified for all eight tasks, but never executed end-to-end against a paid or local judge model.

### Work remaining for Phase 2

1. Build the `legacy` env on the cluster and capture a `legacy.txt` snapshot from it.
2. Run the Phase 2 end-to-end verification block from the plan, including a small-`--limit` eval run.

---

## Phase 3 — Analysis loader — Complete (2026-08-03)

### Why

Eight tasks write eight unrelated output shapes: a dict keyed by sample id, a dict keyed by
sample id whose values are keyed by prompt variant, a flat list, a directory of per-n-gram
files, and — for `neocoder` — a CSV. Every notebook that wanted to compare two models was
re-deriving those shapes by hand. `result_analysis/loader.py` turns all eight into one
long table.

### The two decisions worth reviewing

**The loader flattens, it never aggregates.** One row per scored unit; the notebook does the
mean. An aggregating loader would have had to decide what "the score" of a task is, and
that decision belongs in the analysis, not in the reader.

**A `metric` column was added, so the schema is 8 columns rather than the 7 in the plan.**
Without it a single `eval_score` column silently mixes a DAT semantic distance (~86), an
n-gram coverage in [0, 1], and a binary rubric verdict. Grouping by `metric` is now
mandatory to get a meaningful number, which is the point: `eval_score` is only interpretable
next to `metric`.

### Shape

```
run_id  task  model  sample_id  metric  prompt  output  eval_score
```

`load_records()` returns dicts and has no third-party dependency; `load_outputs()` imports
pandas lazily and returns a DataFrame. Analysis needs pandas, reading does not.

Artifacts are located through `outputs/<label>/<task>/<model>/metadata.json`, and resolved
via symlink → `.path` reference file → recorded native path → the same path re-rooted at the
current repo. The re-rooting is what lets a run produced on the cluster be read on a laptop.

### Bugs found while validating against real artifacts

The parsers were first written from a survey of the evaluation *code*. Checking them against
actual files falsified four of the eight:

| Task | What the code survey implied | What the artifacts actually contain |
|---|---|---|
| `creativity_index` | documents keyed by an `item["dataset"]` field | `dataset` is the constant `"creativity_index"`; the domain is in the `prompt_id` prefix. The join was unnecessary anyway — each evaluation file already copies `prompt` and `response`. |
| `aut` / `ttcw` / `creative_short` | one shared "aut-family" shape | three different shapes: variant-keyed, flat-per-story, and per-story-with-`eval_result` |
| `ttct` | evaluation aligns with inference by index | evaluation drops skipped rows, so indices shift. It also carries no id, so the only stable join key is `(question_type, input.text_cot)`. |
| `neocoder` | evaluation is JSON | the announced evaluation artifact is a **CSV** keyed by `(problem_id, dp)`, and it is shorter than the inference — the evaluator drops rounds it cannot score |

Two further problems were found by re-reading the parsers rather than by running them:
`aut` was putting the *judge's rubric text* in the `prompt` column, which every other task
uses for the prompt given to the model under test; and `creative_math` dropped a criterion
whose verdict block was empty instead of keeping it as an unscored row.

Throughout, a unit that ran but was not scored is kept as a row with `metric = None` and
`eval_score = None`. Dropping it would make the generation count look like the evaluation
count and hide how much the evaluator skipped — for `neocoder` that is 136 of 1194 rounds.

### Verification

`runner/test_loader.sh`, 20 checks. The fixtures are synthetic but their shapes are frozen
from real artifacts, because the local mock judge returns prose and therefore cannot produce
a populated `aut` `cleaned_output`, a non-zero rubric verdict, or a `YES` creative-math
decision. Those are exactly the paths an end-to-end mock run cannot reach.

All eight parsers were also run against real artifacts from the local mock runs: 4040 rows
total, no unexpected nulls.

### Caveat

Every artifact the parsers were validated against was produced by a **mock** endpoint. The
shapes are real, the values are not. A parser that mis-reads a *value* range — as opposed to
a structure — would not be caught by anything run so far.

---

## How to update this file

- When a phase advances, flip its row in the status snapshot and add the relevant date.
- Add a row to the commit ledger for every commit that changes behavior, and update its "Pushed" cell once the push is verified. Record the hash, not the file list — the ledger row for a commit is written in the *following* commit, since a commit cannot contain its own hash.
- Record concrete deviations (paths, line numbers, bug fixes) rather than aspirations — aspirations live in `RESTRUCTURING_PLAN.md`.
- New post-completion bugfixes go under the relevant phase's "Post-completion fixes" subsection.
- Keep this file scannable: tables for status, bullets for facts, no narrative prose.