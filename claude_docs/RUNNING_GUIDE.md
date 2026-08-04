# CreativityPrism — Running Guide

**As of 2026-08-03.** This is the operational entry point for the repository. It is organised by
*who you are*, because the three audiences need almost disjoint subsets of it.

| You are… | Go to | You want to know |
|---|---|---|
| Continuing the restructure — including an agent session on the CRC cluster | [Track A](#track-a--continuing-the-restructure) | what is unproven, how to verify it, how to record it |
| Adding a new task to the benchmark | [Track B](#track-b--adding-a-new-task) | the three files you must write, and the contract each one obeys |
| Benchmarking a new LLM | [Track C](#track-c--benchmarking-a-new-llm) | how to register a model, run it, and read the numbers back without misreading them |

Everyone should read [Chapter 0](#chapter-0--things-everyone-needs) first. It is short.

For *why* the system is shaped this way, and for the list of known bugs and the assumptions the
code makes, see [WORK_SUMMARY.md](WORK_SUMMARY.md).

---

## Chapter 0 — Things everyone needs

### 0.1 The map

```
runner/run.py                 the only entry point
registry/models.yaml          canonical model name → per-bundle alias
registry/tasks/{task}.yaml    which bundle, which adapter, which env
registry/adapters/{task}.sh   translates runner args → the bundle's native CLI
tasks/{bundle}/               the eight original benchmarks, essentially unmodified
outputs/{label}/{task}/{model}/   uniform view: links + metadata.json     (gitignored)
result_analysis/loader.py     every task's output → one long table
claude_docs/                  this guide, the work summary, the plan, the change log
```

Four bundles hold eight tasks:

| Bundle | Tasks |
|---|---|
| `tasks/aut_ttcw_cshort` | `aut`, `ttcw`, `creative_short` |
| `tasks/ttct` | `ttct` |
| `tasks/neocoder_dat` | `neocoder`, `dat` |
| `tasks/math_n_index` | `creative_math`, `creativity_index` |

> **Never run two runs concurrently in the same bundle directory.** Tasks write into shared
> paths inside their bundle and will overwrite each other. `--task all` is sequential and safe;
> two shells are not. Under SLURM this is not a problem — `--task all --slurm` fans out to one
> job per task, and each job's writes are keyed by `--label`.

### 0.2 One-time setup

```bash
git clone <repo-url> && cd creativityprism_v2

bash scripts/setup_envs.sh                    # build all environments
bash scripts/setup_envs.sh --env api          # or just the API-only venv
bash scripts/setup_envs.sh --env modern --prefix /your/scratch/conda_envs
```

`--prefix` is recorded in `registry/environments/.location` (gitignored), so the runner and
adapters find the envs afterwards without being told again.

Three environments exist:

| Env | Backend | Contents | Used by |
|---|---|---|---|
| `modern` | conda (`modern.yml`) | vllm 0.7.2 / torch 2.5.1 | most tasks |
| `legacy` | conda (`legacy.yml`) | vllm 0.5.3.post1 / torch 2.3.1 / py3.11 | `neocoder`, `dat` — that bundle cannot run on the newer vLLM |
| `api` | **venv** | no vLLM, CPU torch only | API-model runs anywhere, including a laptop with no conda and no GPU |

Each task YAML names the env it needs and the runner refuses to start if it is missing.
`setup_envs.sh` picks the backend from the spec file's extension, so `--env api` needs only a
Python ≥ 3.10 on `PATH`.

#### Running without conda or a GPU

vLLM is Linux-only and every open-weight path needs it; the API paths do not.

```bash
bash scripts/setup_envs.sh --env api --prefix /short/path
export CREATIVITYPRISM_FORCE_ENV=api
python runner/run.py --task ttct --model GPT4.1 --judge-model GPT4.1 --label local --limit 2
```

The override applies to every task, so `--task all` works too. Passing an open-weight model
while it is set is rejected up front rather than failing deep inside task code. Unset the
variable to go back to each task's declared env.

> **On Windows use a short `--prefix`** such as `C:/cp-envs`. The default location plus
> `.../site-packages/...` exceeds the 260-character path limit and pip fails partway through
> with an `OSError`.

### 0.3 Credentials

`api_keys.json` at the repo root, gitignored. **It has two shapes at once**, because the bundles
disagree about how to find a key:

```jsonc
{
  // provider block — read by neocoder, dat, creative_math, creativity_index
  "OPENAI_API_KEY":    "...",
  "ANTHROPIC_API_KEY": "...",
  "GENAI_API_KEY":     "...",
  "DEEPSEEK_API_KEY":  "...",

  // model-keyed entries — read by aut / ttcw / creative_short, and by ttct
  "gpt-4.1-2025-04-14": "...",
  "gpt-4.1-mini-2025-04-14": "..."
}
```

`tasks/ttct` looks the key up by the **model alias**, lower-cased
(`all_api_keys[model_name.lower()]`), so a missing alias entry is a `KeyError` and not a helpful
message. Start from the template, which lists every alias the registry can emit:

```bash
cp api_keys.example.json api_keys.json    # then fill in
```

Adapters export the provider block into the environment at run time. Keys already set in your
shell win, and the file is never copied into the repo tree. **Keep it gitignored.**

### 0.4 The runner CLI

```bash
python runner/run.py --list-tasks
python runner/run.py --list-models

python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label my_run
python runner/run.py --task all --model GPT4.1 --judge-model GPT4.1-mini --label my_run
python runner/run.py --config runs/aut_smoke_v2.yaml
```

| Flag | Notes |
|---|---|
| `--task` | a registry name, or `all` |
| `--model` | **canonical** name from `registry/models.yaml`, not a provider ID |
| `--judge-model` | always required; five of eight tasks ignore it (see [Track C](#c3--which-tasks-actually-use-the-judge-model)) |
| `--label` | names the run; becomes the directory under `outputs/`. `--run-id` is a deprecated alias |
| `--limit N` | positive integer; rejected where the task cannot honour it |
| `--inference-only` / `--eval-only` | default is both. There is **no `--mode` flag** |
| `--dry-run` | print the adapter command and stop |
| `--slurm`, `--no-submit`, `--slurm-override k=v` | see [Track A](#a5--slurm) |
| `--config FILE` | YAML or JSON; **mutually exclusive** with the CLI flags above |

`--eval-only` reuses the inference results already stored under the same `--label`, so the label
must match the run you want to score.

A config file needs `task`, `inference_model`, `judge_model`; `label` defaults to the file stem.

```yaml
task: aut
inference_model: GPT4.1
judge_model: GPT4.1-mini
limit: 3
```

### 0.5 Where output goes

Tasks keep writing to their own native locations. On top of that the runner builds a uniform
view:

```
outputs/{label}/{task}/{canonical_model}/
├── inference_output[.json]   → link to the native artifact
├── eval_output[.json]        → link to the per-item evaluation output
└── metadata.json             → models, limit, mode, env, command, exit code, timestamps, native paths
```

Nothing is copied — these are symlinks, so a direct task rerun stays visible through them. Where
symlinks are unavailable (Windows without Developer Mode) the runner writes a
`{kind}_output.path` text file instead and records `link_type: reference`.

> **Analysis code must read `metadata.json` → `artifacts.<kind>.native_path`.** It is populated
> identically on both platforms. Never assume a symlink exists.

`outputs/` is gitignored.

---

## Track A — Continuing the restructure

*For the maintainer, and for agent sessions on the CRC cluster.*

### A1 — Read in this order

1. `claude_docs/RESTRUCTURING_PLAN.md` § **Read before run** — the authoritative list of what
   has and has not actually been executed. Read this before running anything on a new machine.
2. `claude_docs/WORK_SUMMARY.md` — what the system is, what is broken, what the code assumes.
3. `claude_docs/CHANGE_LOG.md` § **Commit ledger** — what landed, when, and whether it is pushed.
4. This guide.

The plan's decisions are the result of design discussion. **Do not re-litigate them without
explicit instruction** — but *do* flag it when reality contradicts one, and say so plainly
rather than quietly working around it.

### A2 — The gates

Five scripts, 92 checks. All of them run on a laptop, none of them need a GPU or a cluster.

```bash
# GOTCHA: leftover overrides silently change what the gates test. Always clear them first.
unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL

for g in test_phase1 test_phase2a test_phase3 test_api_env test_loader; do
  printf "%-16s " "$g"
  bash "runner/$g.sh" >/tmp/g_$g.log 2>&1
  printf "rc=%s  " "$?"
  grep -oE "[0-9]+ passed, [0-9]+ failed" /tmp/g_$g.log | tail -1
done
```

Expect `19 / 4 / 28 / 21 / 20`.

| Gate | What it actually proves |
|---|---|
| `test_phase1.sh` | registry loading, CLI validation, dry-run command construction |
| `test_phase2a.sh` | artifact markers parse, metadata is written, link fallback works |
| `test_phase3.sh` | sbatch generation, three-layer override resolution, `--task all` fan-out, marker durability |
| `test_api_env.sh` | every task module imports under an interpreter with **no vLLM**. This is what catches a regression to a top-level `import vllm`. Skips cleanly if the api venv is absent |
| `test_loader.sh` | all eight parsers against fixtures frozen from real artifact shapes |

**What no gate proves:** that `sbatch` accepts the directives, that the partition names exist,
that the resources are adequate, or that any open-weight model loads. Run one job with `--limit`
before launching a matrix.

### A3 — The cluster checklist

This is the outstanding work, in the order that unblocks the most.

1. **Build the `legacy` conda env.** `legacy.yml` has never been solved anywhere. There is
   deliberately no `legacy.txt` — a `conda list --export` snapshot cannot be authored for an
   environment that has never been built.
2. **Run `neocoder` and `dat` on `legacy`.** Both have only ever run under a forced `api` env.
   Their vllm 0.5.3.post1 / torch 2.3.1 pin is untested.
3. **Submit one real SLURM job.** `--slurm --no-submit` is gated; `sbatch` is not. Partition and
   account names in the template are unverified guesses, so expect to set them.
4. **Run one open-weight model.** Every run to date used a hosted API. This exercises the vLLM
   path, `HF_HOME`, and the SLURM resource defaults simultaneously.
5. **Confirm symlinks.** On Linux `outputs/.../inference_output` should be a real symlink;
   `readlink` on it is the check that could never be run on Windows.
6. **Confirm the `cygpath` branches stay inert.** They are guarded on `uname -s`, so they should
   be, but that branch has never executed on Linux.

Environment knobs that matter on a cluster, all handled in `registry/adapters/_common.sh`:

| Variable | Purpose |
|---|---|
| `CREATIVITYPRISM_ENV_PREFIX` | where the envs live; else `registry/environments/.location` |
| `CREATIVITYPRISM_HF_HOME` | **set this.** HF downloads are tens of GB and must not land in a quota-limited home. Else `registry/environments/.hf_home`, else a warned-about fallback |
| `registry/environments/.cluster_env.sh` | gitignored, site-local. `module load`, `LD_LIBRARY_PATH`, scheduler exports — sourced before env activation. Template: `cluster_env.sh.example` |
| `CREATIVITYPRISM_FORCE_ENV` | overrides every task's declared env |
| `CREATIVITYPRISM_GLOVE_PATH` | point at an existing GloVe copy instead of downloading 2 GB |

### A4 — Landmines

- **`unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL`** before gates and before a real run. A
  leftover base URL is the kind of thing that produces a plausible-looking wrong result.
- **Never run two runs concurrently in the same bundle directory** (see [0.1](#01-the-map)).
- **`neocoder` executes model-generated Python.** The subprocess is not a sandbox. Run it only
  where that is acceptable.
- **`cmd | tail -N` buffers the entire output until the command completes.** For a long run,
  redirect to a log file and read the file.
- **In bash, `!!!` triggers history expansion.** It will mangle a commit message.
- **`.ipynb` line numbers differ between tools.** A rendered notebook view and a raw-JSON grep
  disagree. Read notebook cells through `json.load` when precision matters.
- **`../creativityprism_v2/` is a read-only migration source.** Never modify it.

### A5 — SLURM

```bash
python runner/run.py --task aut --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm
python runner/run.py --task aut --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm --no-submit
```

Scripts land in `slurm_scripts/{label}/{task}_{model}.sbatch`, logs in
`slurm_scripts/{label}/logs/`. Both gitignored. `--task all` fans out to one script and one
queue slot per task, so a slow task does not hold up the rest.

The generated script re-invokes `runner/run.py` **inside** the job with the same arguments minus
the SLURM ones, so artifact linking and `metadata.json` happen on the compute node and a cluster
run and a laptop run produce identical `outputs/` trees. Nothing absolute is baked in: the
script resolves the repo root from its own location and calls `python3` from `PATH`.

Directives resolve in three layers, each overriding the previous:

1. `runner/slurm_template.sbatch` — defaults
2. the `slurm:` block in `registry/tasks/{name}.yaml`
3. `--slurm-override key=value`

An empty value **drops** the directive, which is how a GPU job becomes a CPU one:

```bash
# API models never touch the GPU, so do not queue for one
python runner/run.py --task all --model GPT4.1 --judge-model GPT4.1 --label v3 \
    --slurm --slurm-override gres= --slurm-override partition=smp
```

### A6 — Recording work

After a change that alters behaviour:

1. Add a section to `claude_docs/CHANGE_LOG.md` — *why*, *what was built*, *deviations from the
   plan*, *verification*. Record deviations honestly; a silent deviation is how the plan's status
   lines went stale.
2. Add a row to the **commit ledger** with the gate counts at commit time and whether it is
   pushed.
3. Update the relevant **Status** line in `RESTRUCTURING_PLAN.md`.
4. Re-run all five gates and quote the numbers.

Prose style in these docs: explain the *reason*, not the diff. A reader can see what changed;
they cannot see why the obvious alternative was rejected.

---

## Track B — Adding a new task

*For an LLM researcher contributing a benchmark.*

You write **three files**. You do not modify the runner.

| File | Purpose |
|---|---|
| `tasks/{your_task}/` | your implementation — self-contained, keeps its own layout |
| `registry/tasks/{your_task}.yaml` | metadata the runner reads |
| `registry/adapters/{your_task}.sh` | translates the runner's arguments into your CLI |

Plus **one function** in `result_analysis/loader.py` if you want your results in the shared
table — which you do.

### B1 — The task YAML

```yaml
# yaml-language-server: $schema=./_task.schema.json
name: my_new_task                          # must match the filename and --task
display_name: "My New Task"
domain: divergent_thinking                 # reporting label only
folder: tasks/my_new_task
adapter: registry/adapters/my_new_task.sh
environment: modern                        # modern | legacy
description: "What this task measures."
limit_supported: true                      # absent means false; the runner then rejects --limit

# Optional per-task SBATCH overrides. Leave commented out unless you know the cluster.
# slurm:
#   partition: gpu
#   time: "2:00:00"
#   gres: "gpu:1"
```

`_task.schema.json` sits beside it and is enforced by your editor. `additionalProperties` is
false, so a typo is caught rather than ignored.

Set `limit_supported: false` if your task has no sample-count knob. The runner will then reject
`--limit` with a clear error instead of silently ignoring it, and will reject
`--task all --limit N` outright.

### B2 — The adapter

A ~30-line bash script. The contract:

```bash
#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"          # sets MODEL JUDGE RUN_ID OUTPUT_DIR LIMIT MODE
activate_env modern              # honours CREATIVITYPRISM_FORCE_ENV

ALIAS="$(lookup_alias "$MODEL" my_new_task)"      # canonical → your bundle's spelling
API_ID="$(lookup_alias "$MODEL" api_call 2>/dev/null || echo "$ALIAS")"

TASK_DIR="$REPO_ROOT/tasks/my_new_task"
cd "$TASK_DIR"
NATIVE_OUT="$TASK_DIR/output/${RUN_ID}/${ALIAS}"

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    python my_inference.py --model "$API_ID" --out "$NATIVE_OUT" ${LIMIT:+--limit "$LIMIT"}
    emit_artifact inference "$NATIVE_OUT"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    JUDGE_ID="$(lookup_alias "$JUDGE" api_call 2>/dev/null || lookup_alias "$JUDGE" my_new_task)"
    python my_eval.py --judge "$JUDGE_ID" --in "$NATIVE_OUT"
    emit_artifact eval "$NATIVE_OUT/eval_output.json"
fi
```

Helpers available from `_common.sh`:

| Helper | Does |
|---|---|
| `parse_adapter_args "$@"` | sets `MODEL`, `JUDGE`, `RUN_ID`, `OUTPUT_DIR`, `LIMIT`, `MODE` (`both`\|`inference`\|`eval`) |
| `activate_env <short>` | prepends the env's `bin/` to `PATH`; handles venv vs conda, `HF_HOME`, caches, `CREATIVITYPRISM_FORCE_ENV` |
| `lookup_alias <canonical> <key>` | reads `registry/models.yaml`; **fails loudly** if the alias is absent |
| `emit_artifact <inference\|eval> <abs-path>` | announces an artifact |
| `add_pythonpath <dir>` | makes your bundle importable, handling Windows path form and separator |
| `export_provider_keys` | resolves `OPENAI_API_KEY` etc. from the credentials file into the environment |

Rules that are not optional:

- **Emit a marker only for a phase that actually ran and succeeded.** An `--eval-only`
  invocation must not announce an inference artifact. The runner cannot check this — it never
  reads artifact contents.
- **Paths must be absolute.** `emit_artifact` converts them for Windows consumers itself.
- **The path may be a file or a directory.** Both are linked.
- **`--eval-only` must find its input from `RUN_ID` alone**, since that is all the user passes.

### B3 — Registering your model aliases

Add an alias key for your bundle to each model in `registry/models.yaml`:

```yaml
  GPT4.1:
    type: api
    provider: openai
    aliases:
      api_call: gpt-4.1-2025-04-14
      my_new_task: gpt-4.1                 # ← your bundle's spelling
```

A **missing alias is intentional** — it means the bundle does not support that model, and
`lookup_alias` fails with an explicit message instead of guessing.

### B4 — The loader parser

Write one function and register it. It receives whatever `load_artifact` produced from your
announced artifacts — a list of `(name, data)` pairs — and returns a list of dicts.

```python
def _parse_my_new_task(inference, evaluation):
    """One row per (sample, metric)."""
    rows = []
    for _name, data in inference:
        for item in data:
            rows.append({
                "sample_id": str(item["id"]),
                "metric": "my_metric",          # None if generated but unscored
                "prompt": item.get("prompt"),
                "output": item.get("response"),
                "eval_score": _as_number(item.get("score")),
            })
    return rows

PARSERS["my_new_task"] = _parse_my_new_task     # in the PARSERS dict near the bottom
```

Three conventions the existing parsers all follow, and yours should too:

1. **Flatten, never aggregate.** One row per scored unit. Taking the mean is the notebook's job,
   because what counts as "the score" of a task is an analysis decision.
2. **`metric` must name the unit `eval_score` is in.** A semantic distance, a coverage fraction
   and a binary verdict are not comparable; the column is what stops them being averaged
   together. Where a task produces several numbers per item, emit several rows — never a
   composite.
3. **Keep a generated-but-unscored unit** as a row with `metric = None` and `eval_score = None`.
   The row count should reflect what *ran*, not what the evaluator managed to score. This is
   precisely what made a silent evaluator failure visible in `neocoder`.

Also: **join on keys, never on position.** Evaluation artifacts are routinely shorter than
inference artifacts because the evaluator drops what it could not score. And make sure your join
key is genuinely unique — a non-unique key silently halved `creative_math`'s reported
correctness.

### B5 — Checklist before you open the PR

```bash
python runner/run.py --list-tasks                       # your task appears
python runner/run.py --task my_new_task --model GPT4.1 --judge-model GPT4.1-mini \
       --label check --dry-run                          # command looks right
bash -n registry/adapters/my_new_task.sh                # adapter parses

python runner/run.py --task my_new_task --model GPT4.1 --judge-model GPT4.1-mini \
       --label check --limit 2                          # real, tiny
cat outputs/check/my_new_task/GPT4.1/metadata.json      # artifacts recorded, exit_code 0

python -c "import sys; sys.path.insert(0,'result_analysis'); import loader; \
           print(len(loader.load_records('check')))"    # your rows come back

unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL
bash runner/test_phase1.sh && bash runner/test_api_env.sh && bash runner/test_loader.sh
```

`test_api_env.sh` will fail if any module you added imports vLLM at the top level. Import it
inside the function that needs it.

---

## Track C — Benchmarking a new LLM

*For anyone who wants numbers for a model that is not yet in the registry.*

### C1 — Registering the model you want to test

**This is the first step and the most common first failure.** The runner only accepts canonical
names from `registry/models.yaml`, and each bundle needs its own alias.

```yaml
  My-New-Model:
    type: api                    # api | open
    provider: openai             # openai | anthropic | google | deepseek | huggingface
    # hf_id: org/repo            # open models only
    aliases:
      api_call: my-model-2026-01-01     # the exact ID in API requests
      aut_ttcw_cshort: my_new_model     # folder/alias in that bundle
      ttct: my-model-2026-01-01         # -model_name for ttct_inference.py
      neocoder_dat: my-new-model        # must be a key of __MODEL_TO_CLASS__ in that bundle
      math_n_index: my-new-model        # must be a key of open_source_models / closed_source_model
```

Omit the alias for any bundle that cannot serve the model; `lookup_alias` will then fail with an
explicit message rather than guessing. For `ttct` you must **also** add the alias, lower-cased,
as a key in `api_keys.json`.

> ### The registry currently lists model IDs that no longer serve
>
> Probed directly with this repo's keys on **2026-08-03**:
>
> | Registry name | `api_call` | Status |
> |---|---|---|
> | `GPT4.1` | `gpt-4.1-2025-04-14` | works |
> | `GPT4.1-mini` | `gpt-4.1-mini-2025-04-14` | works |
> | `Claude3-Sonnet` | `claude-3-7-sonnet-20250219` | **404 — retired** |
> | `Claude3-Haiku` | `claude-3-haiku-20240307` | **404 — retired** |
> | `Gemini2.0-Flash` | `gemini-2.0-flash` | **404 — retired** |
> | `Gemini2.0-Pro` | `gemini-2.0-pro` | unverified |
> | `DeepSeek-R1` / `-V3` | `deepseek-reasoner` / `-chat` | unreachable from the verification network (TLS handshake failure) |
>
> **Of the eight API models registered, only two are known to serve.** Current working IDs as of
> the same date: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o-mini`; `claude-sonnet-4-5-20250929`,
> `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`; `gemini-2.5-flash` — **only** with
> `thinkingBudget=0` (see [C6](#c6--traps-that-produce-plausible-wrong-numbers)).
>
> **Listing ≠ serving.** `gemini-2.0-flash` still appears in `GET /v1beta/models` while refusing
> to serve. Do not use the listing endpoint as an availability check; send a real one-token
> request. A dead *key* returns 401; a dead *model* returns a structured 404.

### C2 — Smoke, then run

```bash
unset CREATIVITYPRISM_FORCE_ENV OPENAI_BASE_URL      # start from a clean environment

# 1. Does the plumbing work at all?
python runner/run.py --task aut --model My-New-Model --judge-model GPT4.1-mini \
       --label smoke --limit 2

# 2. Everything, one task at a time (sequential, safe)
python runner/run.py --task all --model My-New-Model --judge-model GPT4.1-mini --label v1

# 3. On a cluster, one job per task
python runner/run.py --task all --model My-New-Model --judge-model GPT4.1-mini --label v1 --slurm
```

Do the `--limit 2` smoke test for **every** task before a full matrix. Half the failures in this
repository's history were a model ID, an alias or a key — all of which surface in the first two
samples and none of which are worth discovering an hour in.

### C3 — Which tasks actually use the judge model

It is always required. Five tasks ignore it.

| Task | Uses `--judge-model`? |
|---|---|
| `aut`, `ttcw`, `ttct` | **yes** — it selects the LLM judge |
| `creative_short` | no — automatic metrics only |
| `creativity_index` | no — exact n-gram overlap |
| `dat` | no — mean pairwise GloVe distance |
| `creative_math` | no — a fixed three-judge panel votes by majority |
| `neocoder` | no — technique detection hardcodes `gpt-4-turbo` |

**Hold the judge constant across every model you intend to compare.** Changing the judge changes
the scale, not just the noise.

### C4 — Cost and prerequisites, per task

| Task | Full-run size | Needs |
|---|---|---|
| `aut`, `ttcw`, `creative_short` | dataset-sized | judge calls for `aut`/`ttcw` |
| `ttct` | **500 items** — 5 scored question types × 100, one generation each | judge calls |
| `creative_math` | dataset × `k` variants | a **cleaning** pass first (see below) |
| `creativity_index` | dataset × n-gram sweep 5..12 × 3 domains | **outbound internet** to `api.infini-gram.io` |
| `dat` | `--limit` = repeat count | `glove.840B.300d.txt`, ~2 GB |
| `neocoder` | fixed | one `gpt-4-turbo` call per solution, **and it executes model-generated Python** |

**`dat`'s GloVe download** is not redistributed:

```bash
cd tasks/neocoder_dat && mkdir -p embeddings/glove
curl -L -o /tmp/glove.zip https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip /tmp/glove.zip -d embeddings/glove
```

`embeddings/` is gitignored; `CREATIVITYPRISM_GLOVE_PATH` points at an existing copy. Only DAT
*evaluation* needs it.

**`creativity_index` is expensive and rate-limited.** Cheaper run:

```bash
export CREATIVITYPRISM_INDEX_MIN_NGRAM=5
export CREATIVITYPRISM_INDEX_DOMAINS=poem
```

**`creative_math` needs a cleaner.** `creative_math_eval_api.py` scores
`sample["cleaned_response"]`, produced by a separate pass that strips novelty commentary so the
correctness judges see the mathematics alone. The adapter runs it at the start of eval and skips
it when the field is already present, so re-running eval costs nothing extra. It defaults to
vLLM + Llama-3.3-70B on 4 GPUs — the published setup. Without GPUs:

```bash
export CREATIVITYPRISM_MATH_CLEANER_BACKEND=openai      # default cleaner: gpt-4o-mini
```

Use the **same** setting for every model in a comparison. Each item records `cleaner_model`, so a
mixed set is detectable after the fact. **Scores cleaned by anything other than Llama-3.3-70B are
not comparable to the published numbers.**

### C5 — Reading the results

```python
# from the repo root
from result_analysis.loader import load_records, load_outputs, list_runs
# from a notebook inside result_analysis/
import loader

list_runs()                     # labels present under outputs/
rows = load_records("v1")       # list of dicts, no pandas needed
df   = load_outputs("v1")       # the same rows as a DataFrame
df   = load_outputs("v1", task="aut", model="My-New-Model")
```

Columns: `run_id, task, model, sample_id, metric, prompt, output, eval_score`.

**Always group by `metric` before averaging.**

```python
df.groupby(["task", "metric"])["eval_score"].mean()
```

`eval_score` alone mixes a DAT distance around 86, a coverage fraction in [0, 1] and a binary
verdict. What each metric means:

| Task | `metric` values | Scale |
|---|---|---|
| `aut` | `novelty` | judge rating |
| `ttcw` | `rubric_q{n}` | binary per rubric question |
| `creative_short` | `dsi`, `surprise`, `n_gram_diversity_{1..5}` | automatic, each on its own scale |
| `ttct` | `fluency`, `flexibility`, `originality`, `elaboration` | judge rating |
| `creative_math` | `correctness`, `coarse_grained_novelty`, `fine_grained_novelty` | binary, panel majority |
| `creativity_index` | `coverage_exact_{5..12}` | fraction in [0, 1], falls as n grows |
| `neocoder` | `correctness`, `follow_constraints`, `new_techniques_ratio` | per denial round |
| `dat` | `dat_score` | mean pairwise GloVe distance, ~86 |

**Rows with `metric = None` were generated but never scored.** They are kept deliberately, so the
row count reflects what ran. Filter them out explicitly when you want only scored rows — and if
there are a lot of them, find out why before averaging anything.

A run produced on a cluster can be read on a laptop: absolute paths in `metadata.json` are
re-rooted at the current repo when the original location does not exist.

### C6 — Traps that produce plausible wrong numbers

Ordered by how quietly they fail.

1. **A dead judge in `creative_math` votes `NO`.** Three of four wrappers return the exception as
   a string, which is then compared against `"YES"`. Correctness requires unanimity, so one
   unavailable provider drives correctness to 0% — and cascades into both novelty grades. **A
   `creative_math` correctness of 0 means "check your keys" until proven otherwise.**
2. **Gemini 2.5+ returns nothing unless thinking is disabled.** With a small
   `max_output_tokens` the response has `finishReason=MAX_TOKENS` and no `parts` key at all.
   Only `tasks/math_n_index/api_warpper.py` is fixed. `aut_ttcw_cshort`, `ttct` and
   `neocoder_dat` are **not** — point Gemini at those and fix them first, or they silently
   return empty text.
3. **`neocoder` correctness is a lower bound of unknown tightness.** Its comparison layer
   exact-matches one reference answer for problems that accept many, and at least one inspected
   "failure" was a correct answer. See [WORK_SUMMARY.md § 2.1](WORK_SUMMARY.md#21-bugs-that-change-published-numbers).
4. **`creativity_index` published before 2026-08-03 is suspect.** Lost API lookups used to be
   scored identically to "n-gram not in corpus", which *raises* the reported index.
5. **A run that used `CREATIVITYPRISM_MATH_CLEANER_BACKEND=openai` is not comparable to the
   paper.** Check `cleaner_model` in the items.
6. **`--eval-only` scores whatever is under that `--label`.** Reusing a label across two models
   silently mixes runs.

---

## Appendix — Document map

| File | What it is |
|---|---|
| [WORK_SUMMARY.md](WORK_SUMMARY.md) | Standalone summary: architecture, every bug found, every decision and its rejected alternative, and what the fixed code assumes. |
| [RESTRUCTURING_PLAN.md](RESTRUCTURING_PLAN.md) | The design contract and phase roadmap. Its "Read before run" section is authoritative on what is unverified. |
| [CHANGE_LOG.md](CHANGE_LOG.md) | Chronological record and commit ledger. |
| [WORKFLOW.md](WORKFLOW.md) | The original community-facing workflow description; superseded operationally by this guide. |
