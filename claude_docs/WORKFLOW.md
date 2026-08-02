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

### Setup (one-time)

```bash
git clone <repo-url>
cd creativityprism_v2

# Install the conda environment(s)
bash scripts/setup_envs.sh

# (Optional) Use a custom install location for conda envs
bash scripts/setup_envs.sh --prefix /your/storage/path/conda_envs
```

There are two environments. `modern` (vllm 0.7.2 / torch 2.5.1) serves most tasks;
`legacy` (vllm 0.5.3.post1 / torch 2.3.1 / Python 3.11) is required by `neocoder` and
`dat` because that bundle cannot run on the newer vLLM. Each task YAML names the one it
needs, and the runner refuses to start if it is missing.

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

### Planned: Pitt CRC submission (Phase 3)

The following is the target interface for users with access to the University of Pittsburgh's Center for Research Computing. `--slurm` and `--no-submit` are not implemented yet.

```bash
# Submit as a SLURM job
python runner/run.py --task aut --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm

# Generate the SLURM script without submitting (for review)
python runner/run.py --task aut --model Qwen2.5-72B --judge-model GPT4.1-mini --label v3 --slurm --no-submit
```

The planned task metadata will provide sensible SLURM defaults (partition, GPU count, time limit, memory) that can be overridden.

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