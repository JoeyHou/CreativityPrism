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
| NeoCoder | Logical reasoning | Planned (Phase 2) |
| DAT (Divergent Association Task) | Divergent thinking | Planned (Phase 2) |
| Creative Math | Logical reasoning | Planned (Phase 2) |
| Creativity Index | Divergent thinking | Planned (Phase 2) |

---

## For Benchmark Users (Running Existing Tasks)

### What you need

1. **A model to evaluate** — either an API-based model (OpenAI, Anthropic, Google) or an open-weight model (via vLLM).
2. **API keys** (for API models) — placed in a local `api_keys.json` file.
3. **GPU access** (for open-weight models) — at least one GPU with sufficient VRAM.

### Setup (one-time)

```bash
git clone <repo-url>
cd creativityprism_v2

# Install the conda environment(s)
bash scripts/setup_envs.sh

# (Optional) Use a custom install location for conda envs
bash scripts/setup_envs.sh --prefix /your/storage/path/conda_envs
```

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

When provided, `--limit` must be a positive integer. Omit it to run the full dataset. Evaluation dispatch is planned for Phase 2; the current adapters perform inference and leave their eval branches stubbed.

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

Phase 1 outputs remain in each task's native location. The bundled tasks use:

```
tasks/aut_ttcw_cshort/data/output/{label}/{task}/{model_alias}/
```

TTCT uses `tasks/ttct/data/outputs/{label}/{model_alias}.json`. Each adapter prints its native `OUTPUT_PATH` on success.

Phase 2 will add the centralized `outputs/{label}/{task}/{canonical_model}/` view, metadata, and links to native artifacts without duplicating data.

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
3. Prints the output file path when done

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