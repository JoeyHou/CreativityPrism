# CreativityPrism Restructuring — Change Log

A running record of restructuring progress. Companion to [RESTRUCTURING_PLAN.md](RESTRUCTURING_PLAN.md), which is the design source of truth. This file tracks **what has actually been built** and **what is still pending**.

New sessions: read `RESTRUCTURING_PLAN.md` first for design intent, then this file for current state, then verify against the codebase before acting.

---

## Status snapshot (2026-07-22)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Foundation (inference + registry + cleanup) | **Complete + Behavior-gated (2026-07-22)** | 4 tasks wired; 19/19 Phase 1 checks pass |
| Phase 2 — Evaluation + Centralized Outputs | **Not started** | Eval branches stubbed; `outputs/` symlinks pending; neocoder/dat/creative_math/creativity_index not wired |
| Phase 3 — SLURM + Analysis Loader | **Not started** | No `--slurm` flag, no `loader.py` |

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
| 5 | AUT, TTCW, and Creative Short shared `data/output/{run_id}/{alias}` and could overwrite one another | Their adapters now use `data/output/{run_id}/{task}/{alias}` for both logical run IDs and `OUTPUT_PATH` |
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

## Phase 2 — Not started

### Confirmed gaps (verified 2026-05-15)

- **No `outputs/` directory** at the repo root. `runner/run.py:97` still marks `--output-dir` as informational ("reserved for Phase 2 symlinks").
- **Eval branches stubbed in all four adapters.** Example: `registry/adapters/aut.sh:35` reads `# Eval is Phase 2; intentionally not implemented yet.`. The `--eval-only` flag is parsed in `_common.sh:26` but never dispatched to task code.
- **Missing task wiring.** Folders `tasks/math_n_index/` and `tasks/neocoder_dat/` exist, but no YAMLs/adapters for `neocoder`, `dat`, `creative_math`, `creativity_index`.
- **No `legacy` env.** Only `modern.txt`/`modern.yml` are present; `legacy.txt` (vllm 0.5.3, torch 2.3.1) for the `neocoder_dat` bundle is not yet created.
- **No `metadata.json` writer** in the runner.

### Work remaining for Phase 2

1. Add eval support to existing adapters (`aut.sh`, `ttcw.sh`, `creative_short.sh`, `ttct.sh`).
2. Add adapters + YAMLs for `neocoder`, `dat`, `creative_math`, `creativity_index`.
3. Add `registry/environments/legacy.{txt,yml}` and ensure `scripts/setup_envs.sh` handles it.
4. Implement symlink creation in `runner/run.py`: parse `OUTPUT_PATH` from adapter stdout, materialize `outputs/{run_id}/{task}/{canonical_model}/`, write `metadata.json`.
5. Run the Phase 2 verification block from the plan.

---

## Phase 3 — Not started

No `--slurm` flag in the runner, no `runner/slurm_template.sbatch`, no `result_analysis/loader.py`. Picked up only after Phase 2 lands.

---

## How to update this file

- When a phase advances, flip its row in the status snapshot and add the relevant date.
- Record concrete deviations (paths, line numbers, bug fixes) rather than aspirations — aspirations live in `RESTRUCTURING_PLAN.md`.
- New post-completion bugfixes go under the relevant phase's "Post-completion fixes" subsection.
- Keep this file scannable: tables for status, bullets for facts, no narrative prose.