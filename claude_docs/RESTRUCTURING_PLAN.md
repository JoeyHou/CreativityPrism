# CreativityPrism Restructuring Plan

This document is the source of truth for the ongoing effort to restructure the CreativityPrism benchmark codebase. It captures the design rationale, the agreed architecture, and a phased implementation roadmap. New Claude Code sessions working on this project should read this file first.

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

#### Marker durability (important for Phase 3)

Markers are **transient**. The runner captures them from the adapter's stdout pipe while the process is alive; nothing writes the marker lines themselves to disk. This is safe today because:

- Nested processes inherit the adapter's stdout, so a marker emitted by a helper script or a nested `bash` still reaches the runner's pipe.
- The durable record is `metadata.json`, written immediately after the adapter exits.

It breaks under **Phase 3 SLURM submission**, where the adapter runs detached inside a batch job and its stdout goes to a SLURM log file the runner never reads. Phase 3 must therefore either parse the job's stdout log after completion, or have `emit_artifact` additionally append to a marker file inside the run's output directory. Decide this when Phase 3 starts; do not retrofit it now.

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

**Status:** Not started

**Scope:**
- Add `--slurm` flag to the runner.
- Build `result_analysis/loader.py` for unified output loading.

**Runner changes:**
- `--slurm` generates an sbatch script wrapping the same command, then calls `sbatch` (unless `--no-submit`).
- SLURM headers come from `runner/slurm_template.sbatch`, with per-task overrides from `registry/tasks/{name}.yaml`.
- Generated scripts go to `slurm_scripts/{run_id}/{task}_{model}.sbatch`.

**`result_analysis/loader.py`:**

```python
from result_analysis.loader import load_outputs

# Load all outputs for a run
df = load_outputs(run_id="v3")
# → DataFrame: run_id, task, model, sample_id, prompt, output, eval_score

# Filter by task and/or model
df = load_outputs(run_id="v3", task="aut", model="GPT4.1")
```

- One parser function per task (knows the native format).
- Reads from `outputs/{run_id}/...` only, resolving artifacts via `metadata.json` (which records native paths whether or not symlinks were available).
- Reuses canonical model names from `registry/models.yaml`.

**Verification steps:**

```bash
# SLURM dry-run
python runner/run.py --task aut --model GPT4.1 --judge-model GPT4.1-mini --label v3 --slurm --no-submit
cat slurm_scripts/v3/aut_GPT4.1.sbatch   # verify sbatch script is well-formed

# Full SLURM submission (real)
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