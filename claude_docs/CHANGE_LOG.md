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
| Phase 2C — Remaining task adapters | **Not started** | neocoder/dat/creative_math/creativity_index not wired; no `legacy` env |
| Phase 3 — SLURM + Analysis Loader | **Not started** | No `--slurm` flag, no `loader.py` |

---

## Commit ledger

Anchors each phase to the commit that implemented it, so a future session can jump straight to the diff. This table deliberately does **not** duplicate file lists or line numbers — `git show <hash> --stat` and `git show <hash> -- <path>` are the authoritative per-file record and never go stale. What lives here instead is the mapping and the verification state, which git does not capture.

| Commit | Date | Phase | Scope | Gates at commit time | Pushed to `personal/main_v2` |
|--------|------|-------|-------|----------------------|------------------------------|
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
- Next slice: Phase 2C — adapters for the remaining four tasks plus the `legacy` conda env.

See the top of `RESTRUCTURING_PLAN.md` for the new-session startup gate and safety constraints.

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

- **Missing task wiring.** Folders `tasks/math_n_index/` and `tasks/neocoder_dat/` exist, but no YAMLs/adapters for `neocoder`, `dat`, `creative_math`, `creativity_index`.
- **No `legacy` env.** Only `modern.txt`/`modern.yml` are present; `legacy.txt` (vllm 0.5.3, torch 2.3.1) for the `neocoder_dat` bundle is not yet created.
- **No judge has actually been run.** Eval dispatch is wired and statically verified, but never executed end-to-end against a paid or local judge model.

### Work remaining for Phase 2C

1. Add adapters + YAMLs for `neocoder`, `dat`, `creative_math`, `creativity_index`.
2. Add `registry/environments/legacy.{txt,yml}` and ensure `scripts/setup_envs.sh` handles it.
3. Run the Phase 2 end-to-end verification block from the plan, including a small-`--limit` eval run.

---

## Phase 3 — Not started

No `--slurm` flag in the runner, no `runner/slurm_template.sbatch`, no `result_analysis/loader.py`. Picked up only after Phase 2 lands.

---

## How to update this file

- When a phase advances, flip its row in the status snapshot and add the relevant date.
- Add a row to the commit ledger for every commit that changes behavior, and update its "Pushed" cell once the push is verified. Record the hash, not the file list — the ledger row for a commit is written in the *following* commit, since a commit cannot contain its own hash.
- Record concrete deviations (paths, line numbers, bug fixes) rather than aspirations — aspirations live in `RESTRUCTURING_PLAN.md`.
- New post-completion bugfixes go under the relevant phase's "Post-completion fixes" subsection.
- Keep this file scannable: tables for status, bullets for facts, no narrative prose.