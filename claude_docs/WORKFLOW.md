# CreativityPrism: Benchmark Workflow

This document describes how researchers and community contributors interact with the CreativityPrism benchmark.

---

## What is CreativityPrism?

CreativityPrism is a benchmark suite for evaluating the creativity of large language models. It covers three domains — **divergent thinking**, **creative writing**, and **logical reasoning** — across eight tasks, measuring quality, novelty, and diversity of model outputs.

**Benchmark tasks:**

| Task | Domain | Runner status |
|------|--------|---------------|
| AUT (Alternative Uses Task) | Divergent thinking | Runnable (Phase 1) |
| TTCW (Torrance Tests of Creative Writing) | Creative writing | Runnable (Phase 1) |
| Creative Short | Creative writing | Runnable (Phase 1) |
| TTCT (Torrance Tests of Creative Thinking) | Divergent thinking | Runnable (Phase 1) |
| NeoCoder | Logical reasoning | Planned (Phase 2C) |
| DAT (Divergent Association Task) | Divergent thinking | Planned (Phase 2C) |
| Creative Math | Logical reasoning | Planned (Phase 2C) |
| Creativity Index | Divergent thinking | Planned (Phase 2C) |

---

## For Benchmark Users (Running Existing Tasks)

### What you need

1. **A model to evaluate** — either an API-based model (OpenAI, Anthropic, Google) or an open-weight model (via vLLM).
2. **API keys** (for API models) — placed in a local `api_keys.json` file.
3. **GPU access** (for open-weight models) — at least one GPU with sufficient VRAM.

#### `api_keys.json`

The file is model-keyed (`"gpt-4.1": "sk-..."`) for the AUT/TTCW/Creative Short and
TTCT tasks. `neocoder`, `creative_math` and `creativity_index` instead read provider
environment variables, so add a provider block alongside the model entries:

```json
{
  "OPENAI_API_KEY": "...",
  "ANTHROPIC_API_KEY": "...",
  "GENAI_API_KEY": "...",
  "DEEPSEEK_API_KEY": "..."
}
```

Adapters export these into the environment at run time. Keys already set in your shell
take priority, and the file is never copied into the repo tree. `api_keys.json` is
gitignored — keep it that way.

`tasks/ttct` does not read the provider fields; it looks the key up by the **model
alias** (`all_api_keys[model_name.lower()]`). Copy `api_keys.example.json`, which lists
every alias the registry can emit, rather than writing the file from scratch:

```bash
cp api_keys.example.json api_keys.json   # then fill in
```

### Setup (one-time)

```bash
git clone <repo-url>
cd creativityprism_v2

# Install the conda environment(s)
bash scripts/setup_envs.sh

# (Optional) Use a custom install location for conda envs
bash scripts/setup_envs.sh --prefix /your/storage/path/conda_envs
```

There are three environments.

| Env | Backend | Contents | Used by |
|-----|---------|----------|---------|
| `modern` | conda (`modern.yml`) | vllm 0.7.2 / torch 2.5.1 | most tasks |
| `legacy` | conda (`legacy.yml`) | vllm 0.5.3.post1 / torch 2.3.1 / py3.11 | `neocoder`, `dat` — that bundle cannot run on the newer vLLM |
| `api` | **venv** (`api.requirements.txt`) | no vLLM, CPU torch only | API-model runs on any OS, including machines without conda or a GPU |

Each task YAML names the environment it needs, and the runner refuses to start if it is
missing. `scripts/setup_envs.sh` picks the backend from the spec file extension, so
`--env api` needs only a Python ≥ 3.10 on PATH.

#### Running API models without conda or a GPU

vLLM is Linux-only and every open-weight path needs it, but the API paths do not. Set
`CREATIVITYPRISM_FORCE_ENV=api` to run any task against an API model on a laptop:

```bash
bash scripts/setup_envs.sh --env api
export CREATIVITYPRISM_FORCE_ENV=api
python runner/run.py --task ttct --model GPT4.1 --judge-model GPT4.1 --label local --limit 2
```

The override applies to every task, so `--task all` works too. Passing an open-weight
model while it is set is rejected up front rather than failing inside the task code.
The venv lands in `registry/environments/creativityprism-api/` (gitignored) unless
`--prefix` says otherwise. Unset the variable to go back to the task's declared env.

On Windows the default location is usually **too deep**: the repo path plus
`.../site-packages/...` exceeds the 260-character limit and pip fails partway through
with an `OSError`. Install to a short path instead:

```bash
bash scripts/setup_envs.sh --env api --prefix C:/cp-envs
```

The prefix is recorded in `registry/environments/.location` (gitignored), so the runner
and adapters find it automatically afterwards.

`bash runner/test_api_env.sh` guards this path: it imports every task module using the
api venv's interpreter, which has no vLLM. Any module that regresses to a top-level
`import vllm` fails the gate. It skips cleanly when the api venv is absent.

Two Windows details are handled by the setup script and adapters, not by the user:
venvs there ship `Scripts/python.exe` but no `python3.exe` (which every adapter calls),
so a copy is made; and `activate_env` converts the venv path to POSIX form before
prepending it to `PATH`, since a `C:/...` entry would be split at the colon.

Without Developer Mode, Windows cannot create symlinks, so the centralized output dir
gets an `*_output.path` text reference instead of `*_output.json`. `metadata.json`
records this under `artifacts.<kind>.link_type: reference`.

#### `--limit` and the ttct eval phase

ttct's eval phase asserts the inference output has one row per
`data/processed/basefile.csv` row (700), so `--limit N` cannot truncate the file. It keeps
all 700 rows and queries only the **first N items of each scored question type**, marking
the rest `skip` so the judge skips them too. `--task ttct --limit 2 --mode both` therefore
issues 10 inference calls and scores 10 items.

The dataset ships 7 question types but only 5 are scored: the LLM-judge rubric was aligned
against human ratings for `1_unusual_uses`, `2_consequences`, `4_situation`,
`5_common_problems` and `6_improvement` only. `3_just_suppose` and `7_story` ship for
completeness and stay unscored. See `tasks/ttct/README.md`.

#### The `creative_math` cleaning step

`creative_math_eval_api.py` scores `sample["cleaned_response"]`, a field produced only by
`src/utils/clean_data_creative_math.py`, which strips novelty commentary so the correctness
judges see the mathematics alone. The adapter runs it at the start of the eval phase and
skips it when the field is already present, so re-running eval costs nothing extra.

The cleaner is a **fixed instrument, deliberately independent of the model under test**:
letting it track the model being evaluated would make api-model and open-model scores
non-comparable. It defaults to `vllm` with Llama-3.3-70B on 4 GPUs, the published setup.

The api env cannot host that, so `creative_math --eval-only` there exits 4 with instructions
rather than silently substituting a smaller cleaner. To clean via API, set
`CREATIVITYPRISM_MATH_CLEANER_BACKEND=openai` (defaults to `gpt-4o-mini`, override with
`CREATIVITYPRISM_MATH_CLEANER_MODEL`) — and use the **same** setting for every model in the
comparison. Each item records `cleaner_model`, so a mixed set is detectable after the fact.
Scores cleaned by anything other than Llama-3.3-70B are not comparable to the paper.

#### Extra download for `dat`

DAT scoring needs `glove.840B.300d.txt` (~2 GB), which is not redistributed here:

```bash
cd tasks/neocoder_dat
mkdir -p embeddings/glove
curl -L -o /tmp/glove.zip https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip /tmp/glove.zip -d embeddings/glove
```

`embeddings/` is gitignored. Point `$CREATIVITYPRISM_GLOVE_PATH` at an existing copy to
skip the download. Only `dat` evaluation needs it — every other task, and DAT inference,
run without it.

### Running tasks

All tasks are run through a single unified command-line interface:

```bash
# Run a specific task with a specific model
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label my_experiment

# Run all currently registered tasks at once
python runner/run.py --task all --model GPT4.1 --judge-model GPT4.1-mini --label my_experiment

# Run inference only (skip evaluation)
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label my_experiment --inference-only

# Quick smoke test with a small sample
python runner/run.py --task aut --model GPT4.1-mini --judge-model GPT4.1-mini --label test --limit 5

# See available tasks and models
python runner/run.py --list-tasks
python runner/run.py --list-models
```

Users always refer to models by their **canonical name** (e.g., `GPT4.1`, `Qwen2.5-72B`, `Claude3-Sonnet`). The system handles translating these to whatever format each task expects internally.

When provided, `--limit` must be a positive integer. Omit it to run the full dataset.
Not every task supports it: `neocoder` exposes no sample-count knob, so
`--limit` is rejected there with a clear error rather than silently ignored. Because of
that, `--task all --limit N` is rejected outright. For `dat`, `--limit` maps to
`--repeat`: DAT has a single fixed prompt rather than a dataset, so the repeat count *is*
the sample count.

Every run does inference and evaluation by default. Use `--inference-only` or `--eval-only` to run a single phase; `--eval-only` reuses the inference results already stored under the same `--label`, so the label must match the run you want to score. `--judge-model` is always required by the CLI, but several tasks ignore it:

| Task | Does `--judge-model` do anything? |
|------|-----------------------------------|
| `aut`, `ttcw`, `ttct` | Yes — it selects the LLM judge. |
| `creative_short` | No — evaluation is fully automated (no LLM judge). |
| `creativity_index` | No — the metric is exact n-gram overlap, not a judge. |
| `dat` | No — the metric is mean pairwise GloVe distance. |
| `creative_math` | No — a fixed three-judge panel (gpt-4.1, claude-3-7-sonnet, gemini-2.0-flash) votes by majority. |
| `neocoder` | No — technique detection hardcodes `gpt-4-turbo`. |

Two tasks have costs worth knowing before you start them:

- **`neocoder`** executes model-generated Python during correctness scoring, and bills
  one `gpt-4-turbo` call per generated solution during technique detection. Run it only
  where executing untrusted code is acceptable.
- **`creativity_index`** queries `https://api.infini-gram.io/` during evaluation, so the
  compute node needs outbound internet. By default it sweeps `min_ngram` 5..12 across
  all three domains; set `CREATIVITYPRISM_INDEX_MIN_NGRAM=5` and/or
  `CREATIVITYPRISM_INDEX_DOMAINS=poem` for a cheaper run.

### SLURM submission

```bash
# Submit one job per task
python runner/run.py --task aut --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm

# Generate the sbatch script without submitting it, for review
python runner/run.py --task aut --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm --no-submit
```

Scripts land in `slurm_scripts/{label}/{task}_{model}.sbatch`, with job logs in
`slurm_scripts/{label}/logs/`. Both are git-ignored. `--task all` fans out to one
script and one queue slot per task, so a slow task does not hold up the rest.

The generated script re-invokes `runner/run.py` inside the job with the same
arguments minus the SLURM ones. That is deliberate: the runner in the job does the
usual artifact linking and writes `metadata.json`, so a cluster run and a laptop run
produce identical `outputs/` trees and nothing has to be collected afterwards.

Nothing absolute is baked into the script. It resolves the repo root from its own
location and calls `python3` from `PATH`, so a script generated on a laptop still
runs on the cluster.

#### Choosing resources

Directives resolve in three layers, each overriding the one before:

1. `runner/slurm_template.sbatch` — the defaults.
2. The `slurm:` block in `registry/tasks/{name}.yaml`.
3. `--slurm-override key=value` on the command line.

A directive that resolves to an empty value is dropped, which is the way to turn a
GPU job into a CPU-only one:

```bash
# API models never touch the GPU, so do not queue for one
python runner/run.py --task all --model GPT4.1 --judge-model GPT4.1 --label v3 \
    --slurm --slurm-override gres= --slurm-override partition=smp
```

Partition, account and time limits are cluster-specific, so the shipped defaults are
a starting point rather than a recommendation. The `slurm:` blocks in the task files
are commented out for the same reason.

#### What is not covered by the test gate

`runner/test_phase3.sh` checks script generation, override resolution, fan-out and
marker durability, all of which run on a laptop. It cannot check that `sbatch`
accepts the directives, that the partition names exist, or that the requested
resources are enough. Run one job with `--limit` before launching a full matrix.

### Running on your own machine

The implemented non-SLURM `runner/run.py` commands work on Linux machines with conda installed.

### Where outputs go

Tasks keep writing to their own native locations. The bundled tasks use:

```
tasks/aut_ttcw_cshort/data/output/{label}/{task}/{model_alias}/
```

TTCT uses `tasks/ttct/data/outputs/{label}/{model_alias}.json`.

The Phase 2C tasks use:

| Task | Native inference output | Native evaluation output |
|------|-------------------------|--------------------------|
| `neocoder` | `tasks/neocoder_dat/results/{label}/neocoder/inference/` | `.../evaluation/{correctness,creativity}/` |
| `dat` | `tasks/neocoder_dat/results/{label}/dat/inference/` | `.../dat/evaluation/` |
| `creative_math` | `tasks/math_n_index/data/outputs/{label}/creative_math/{alias}/{alias}.json` | `tasks/math_n_index/data/evaluations/{label}/creative_math/` |
| `creativity_index` | `tasks/math_n_index/data/outputs/{label}/creative_index/{alias}/` | `tasks/math_n_index/data/evaluations/{label}/creative_index/{alias}/` |

On top of that, the runner builds a centralized, uniform view of every run:

```
outputs/{label}/{task}/{canonical_model}/
├── inference_output[.json]   # link to the native artifact
├── eval_output[.json]        # link to the per-item evaluation output
└── metadata.json            # models, limit, mode, command, exit code, timestamps, native paths
```

Nothing is copied — these are symlinks, so a direct task rerun stays visible through them. On platforms where symlinks are unavailable (Windows without Developer Mode), the runner writes a `{kind}_output.path` file containing the native path instead. `metadata.json` always records the native path either way, so analysis code can rely on it alone.

For the bundled tasks the evaluator writes its results back into the inference directory, so `eval_output.json` points at `eval_output_cleaned.json` (per-item judge output) and the aggregate `eval_report.csv` sits beside it, reachable through the `inference_output` link.

`outputs/` is gitignored.

### Reading results back

`result_analysis/loader.py` turns every task's output into one long table, so a notebook
does not have to know that `aut` nests responses under a prompt variant while `neocoder`
scores land in a CSV.

```python
# from the repo root
from result_analysis.loader import load_records, load_outputs, list_runs

# from a notebook inside result_analysis/
import loader

list_runs()                              # labels present under outputs/
rows = load_records("v3")                # list of dicts, no pandas needed
df   = load_outputs("v3")                # same rows as a DataFrame
```

Columns are `run_id, task, model, sample_id, metric, prompt, output, eval_score`.

Two things to know before using it:

- **It flattens, it does not aggregate.** One row per scored unit. Taking the mean is
  the notebook's job, because what counts as "the score" of a task is an analysis
  decision.
- **`eval_score` is only meaningful next to `metric`.** One task's score is a semantic
  distance around 86, another's is a coverage fraction in [0, 1], another's is a binary
  verdict. Always group by `metric` (and `task`) before averaging:

  ```python
  df.groupby(["task", "metric"])["eval_score"].mean()
  ```

A unit that was generated but never scored is kept as a row with `metric = None` and
`eval_score = None` rather than dropped, so the row count still reflects what actually
ran. Filter those out explicitly when you want only scored rows.

Artifacts are found through `metadata.json`, and a run produced on the cluster can be read
on a laptop: absolute paths recorded in the metadata are re-rooted at the current repo when
the original location does not exist.

`runner/test_loader.sh` is the gate, 20 checks.

---

## For Community Contributors (Adding a New Task)

### What to submit (via pull request)

Adding a new task requires **three things**:

| File | Purpose |
|------|---------|
| `tasks/{your_task}/` | Your task implementation (code, data, scripts) — self-contained |
| `registry/tasks/{your_task}.yaml` | A short YAML declaring your task's metadata (name, domain, environment) |
| `registry/adapters/{your_task}.sh` | A shell adapter that translates the runner's standard arguments into your task's native commands |

### Task YAML (example)

```yaml
name: my_new_task
display_name: "My New Task"
domain: divergent_thinking
folder: tasks/my_new_task
adapter: registry/adapters/my_new_task.sh
environment: modern
description: "A brief description of what this task measures"
```

### Adapter script

The adapter is a short shell script (~30 lines) that:
1. Receives standardized arguments (`--model`, `--run-id`, etc.)
2. Calls your task's existing code
3. Announces each artifact it produced, using the helper from `registry/adapters/_common.sh`:

```bash
emit_artifact inference "$NATIVE_OUT"   # prints: CP_ARTIFACT inference <path>
emit_artifact eval "$EVAL_OUT"
```

Emit a marker only for a phase that actually ran and succeeded — an `--eval-only` invocation must not announce an inference artifact. The path may be a file or a directory. The runner links it into `outputs/` and records it in `metadata.json`; it never reads or interprets the contents.

Contributors can write their task code in **any language** — Python, R, Julia, etc. The adapter is the only required glue.

### Environment

If your task works with the existing conda environments, just declare which one in your YAML. If it needs a new environment, add a `registry/environments/{name}.txt` requirements file.

### No central code changes needed

The runner and registry are designed so that **no existing files need to be modified** to add a new task. Drop your folder, add two files to `registry/`, and open a PR.

---

## Architecture at a Glance

```
                    User
                      |
                      v
              runner/run.py          (unified CLI — task-agnostic)
                      |
            +---------+---------+
            |         |         |
            v         v         v
        adapter    adapter    adapter   (per-task shell scripts in registry/)
            |         |         |
            v         v         v
        tasks/a    tasks/b    tasks/c   (self-contained task implementations)
            |         |         |
            v         v         v
        outputs/  (centralized symlinks for easy analysis)
```

The runner is intentionally "dumb" — it reads task metadata from YAML files, invokes the right adapter, and organizes outputs. All task-specific logic stays inside the task folder.