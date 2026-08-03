#!/usr/bin/env python3
"""CreativityPrism unified runner.

Reads registry/tasks/*.yaml and registry/models.yaml, builds an adapter
invocation, and runs (or prints) it.

Phase 2.1a: --config support, mandatory judge model, --label.
Phase 2A: artifact markers are parsed from adapter stdout and materialized
under outputs/{label}/{task}/{model}/ together with metadata.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

try:
    from runner import artifacts
    from runner import slurm as slurm_mod
except ImportError:  # executed as a script: runner/ is already on sys.path
    import artifacts
    import slurm as slurm_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "registry"
TASKS_DIR = REGISTRY / "tasks"
MODELS_FILE = REGISTRY / "models.yaml"
LOCATION_FILE = REGISTRY / "environments" / ".location"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_API_KEYS = REPO_ROOT / "api_keys.json"

REQUIRED_CONFIG_FIELDS = ("task", "inference_model", "judge_model")


# ---------- Loaders ----------

def load_tasks():
    tasks = {}
    if not TASKS_DIR.is_dir():
        return tasks
    for f in sorted(TASKS_DIR.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh) or {}
        name = data.get("name") or f.stem
        tasks[name] = data
    return tasks


def load_models():
    if not MODELS_FILE.is_file():
        return {}
    with open(MODELS_FILE) as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("models", {})


def load_run_config(path):
    """Load a run config from YAML or JSON. Returns a dict."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        sys.stderr.write(f"Config file not found: {cfg_path}\n")
        sys.exit(5)
    with open(cfg_path) as f:
        suffix = cfg_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            sys.stderr.write(
                f"Unsupported config format '{suffix}'. Use .yaml/.yml or .json.\n"
            )
            sys.exit(5)
    if not isinstance(data, dict):
        sys.stderr.write(f"Config file {cfg_path} must be a mapping at the top level.\n")
        sys.exit(5)
    return data


def validate_run_config(cfg, tasks, models, source_label):
    """Validate a resolved run config. Exits non-zero on error."""
    errors = []
    for field in REQUIRED_CONFIG_FIELDS:
        if field not in cfg or cfg[field] in (None, ""):
            errors.append(f"missing required field '{field}'")

    if errors:
        for e in errors:
            sys.stderr.write(f"Config error ({source_label}): {e}\n")
        sys.exit(5)

    task = cfg["task"]
    inf_model = cfg["inference_model"]
    judge_model = cfg["judge_model"]

    if task != "all" and task not in tasks:
        avail = ", ".join(sorted(tasks)) or "(none)"
        errors.append(f"unknown task '{task}'. Available: {avail}")
    if inf_model not in models:
        errors.append(
            f"unknown inference_model '{inf_model}'. Run: python runner/run.py --list-models"
        )
    if judge_model not in models:
        errors.append(
            f"unknown judge_model '{judge_model}'. Run: python runner/run.py --list-models"
        )

    limit = cfg.get("limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            errors.append("'limit' must be a positive integer")
        elif task == "all":
            unsupported = sorted(
                name for name, meta in tasks.items()
                if not meta.get("limit_supported", False)
            )
            if unsupported:
                errors.append(
                    "limit is not supported by task(s): " + ", ".join(unsupported)
                )
        elif task in tasks and not tasks[task].get("limit_supported", False):
            errors.append(f"limit is not supported by task '{task}'")

    if errors:
        for e in errors:
            sys.stderr.write(f"Config error ({source_label}): {e}\n")
        sys.exit(5)


# ---------- Pre-flight ----------

def env_exists(env_short):
    """Check whether the env for `env_short` exists, as a venv or a conda env."""
    env_name = f"creativityprism-{env_short}"
    prefix = os.environ.get("CREATIVITYPRISM_ENV_PREFIX")
    if prefix is None and LOCATION_FILE.is_file():
        prefix = LOCATION_FILE.read_text().strip()

    venv_dir = Path(prefix) / env_name if prefix else LOCATION_FILE.parent / env_name
    if (venv_dir / "bin" / "python").exists() or (venv_dir / "Scripts" / "python.exe").exists():
        return True

    if prefix:
        return (Path(prefix) / env_name).is_dir()

    if not shutil.which("conda"):
        return False
    try:
        out = subprocess.check_output(["conda", "env", "list"], text=True)
    except subprocess.CalledProcessError:
        return False
    for line in out.splitlines():
        if line.strip().startswith(env_name + " ") or line.strip() == env_name:
            return True
    return False


def check_env_model_compat(models=None, model_names=()):
    """Pure config validation, so it runs during --dry-run too."""
    forced = os.environ.get("CREATIVITYPRISM_FORCE_ENV")
    if not forced:
        return
    # The API-only env has no vLLM, so an open-weight model would fail deep inside
    # the task code. Reject it here instead.
    for name in model_names:
        if (models or {}).get(name, {}).get("type") == "open":
            sys.stderr.write(
                f"CREATIVITYPRISM_FORCE_ENV={forced} but '{name}' is an open-weight "
                f"model, which needs vLLM and a GPU. Use an api-type model "
                f"(python runner/run.py --list-models) or unset the override.\n"
            )
            sys.exit(5)


def preflight(task_meta):
    env_short = os.environ.get(
        "CREATIVITYPRISM_FORCE_ENV", task_meta.get("environment", "modern")
    )
    if not env_exists(env_short):
        sys.stderr.write(
            f"Environment 'creativityprism-{env_short}' not found. "
            f"Run: bash scripts/setup_envs.sh --env {env_short}\n"
        )
        sys.exit(4)


# ---------- Command builder ----------

def build_adapter_command(task_meta, inference_model, judge_model, label, limit, mode):
    adapter_rel = task_meta.get("adapter")
    if not adapter_rel:
        raise ValueError(f"Task '{task_meta.get('name')}' missing 'adapter' in YAML")
    adapter_path = (REPO_ROOT / adapter_rel).resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    cmd = [
        "bash", str(adapter_path),
        "--model", inference_model,
        "--judge-model", judge_model,
        "--run-id", label,
    ]
    out_dir = artifacts.centralized_output_dir(
        OUTPUTS_ROOT, label, task_meta["name"], inference_model
    )
    cmd += ["--output-dir", str(out_dir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if mode == "inference":
        cmd += ["--inference-only"]
    elif mode == "eval":
        cmd += ["--eval-only"]
    return cmd


def run_adapter(cmd, env):
    """Run the adapter, echoing stdout live while capturing it for markers."""
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, text=True, errors="replace", bufsize=1
    )
    captured = []
    with proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
    return proc.wait(), captured


# ---------- CLI ----------

def cmd_list_tasks(tasks):
    if not tasks:
        print("(no tasks registered)")
        return
    width = max(len(n) for n in tasks)
    for name, meta in sorted(tasks.items()):
        print(f"  {name:<{width}}  {meta.get('display_name', '')}")


def cmd_list_models(models):
    if not models:
        print("(no models registered)")
        return
    width = max(len(n) for n in models)
    for name, meta in sorted(models.items()):
        kind = meta.get("type", "?")
        provider = meta.get("provider", "")
        print(f"  {name:<{width}}  [{kind}] {provider}")


def resolve_invocation(args, tasks, models):
    """Merge --config and CLI args into a single resolved invocation dict.

    Returns: { task, inference_model, judge_model, label, limit }
    Exits non-zero on validation failure.
    """
    cli_flags_set = any([args.task, args.model, args.judge_model,
                         args.label, args.run_id, args.limit is not None])

    if args.config:
        if cli_flags_set:
            sys.stderr.write(
                "--config is mutually exclusive with "
                "--task/--model/--judge-model/--label/--run-id/--limit\n"
            )
            sys.exit(5)
        cfg = load_run_config(args.config)
        validate_run_config(cfg, tasks, models, source_label=args.config)
        label = cfg.get("label") or Path(args.config).stem
        return {
            "task": cfg["task"],
            "inference_model": cfg["inference_model"],
            "judge_model": cfg["judge_model"],
            "label": label,
            "limit": cfg.get("limit"),
        }

    # CLI path
    missing = []
    if not args.task:
        missing.append("--task")
    if not args.model:
        missing.append("--model")
    if not args.judge_model:
        missing.append("--judge-model")
    if args.label and args.run_id:
        sys.stderr.write(
            "--label and --run-id are mutually exclusive (--run-id is deprecated; prefer --label)\n"
        )
        sys.exit(5)
    label = args.label or args.run_id
    if not label:
        missing.append("--label")
    if missing:
        sys.stderr.write(
            "Missing required argument(s): " + ", ".join(missing) +
            "\nUse --config <file> to submit a run config, or pass all CLI args.\n"
        )
        sys.exit(5)

    cfg = {
        "task": args.task,
        "inference_model": args.model,
        "judge_model": args.judge_model,
        "limit": args.limit,
    }
    validate_run_config(cfg, tasks, models, source_label="CLI args")
    return {
        "task": args.task,
        "inference_model": args.model,
        "judge_model": args.judge_model,
        "label": label,
        "limit": args.limit,
    }


def submit_slurm_jobs(args, inv, tasks, selected):
    """Render one sbatch script per selected task, and submit unless --no-submit."""
    try:
        overrides = slurm_mod.parse_overrides(args.slurm_override)
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 5

    exit_code = 0
    for tname in selected:
        try:
            script_path, _content = slurm_mod.build_job(
                REPO_ROOT, tname, tasks[tname], inv["inference_model"],
                inv["label"], sys.argv[1:], overrides,
            )
        except (OSError, ValueError) as e:
            sys.stderr.write(f"[{tname}] could not build sbatch script: {e}\n")
            exit_code = 6
            continue

        rel = script_path.relative_to(REPO_ROOT)
        if args.no_submit:
            print(f"[{tname}] wrote {rel} (not submitted)")
            continue
        job_id, message = slurm_mod.submit(script_path)
        if job_id is None:
            sys.stderr.write(f"[{tname}] submission failed: {message}\n")
            exit_code = 7
        else:
            print(f"[{tname}] submitted {rel} as job {job_id}")
    return exit_code


def main():
    p = argparse.ArgumentParser(
        prog="run.py",
        description="CreativityPrism unified runner.",
    )
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--config", help="Path to a run config file (YAML or JSON). Mutually exclusive with CLI args.")
    p.add_argument("--task", help="Task name (or 'all')")
    p.add_argument("--model", help="Canonical inference model name (see --list-models)")
    p.add_argument("--judge-model", dest="judge_model",
                   help="Canonical judge model name (required)")
    p.add_argument("--label", help="Human-readable run label (used in output paths)")
    p.add_argument("--run-id", help="[Deprecated alias for --label]")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples (must be a positive integer)")
    p.add_argument("--inference-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the adapter command(s) but do not execute")
    p.add_argument("--slurm", action="store_true",
                   help="Generate an sbatch script per task and submit it")
    p.add_argument("--no-submit", action="store_true",
                   help="With --slurm, write the sbatch script(s) but do not call sbatch")
    p.add_argument("--slurm-override", action="append", metavar="KEY=VALUE", default=[],
                   help="Override an SBATCH directive, e.g. time=8:00:00. "
                        "An empty value drops it: gres=")
    args = p.parse_args()

    tasks = load_tasks()
    models = load_models()

    if args.list_tasks:
        cmd_list_tasks(tasks)
        return 0
    if args.list_models:
        cmd_list_models(models)
        return 0

    if args.inference_only and args.eval_only:
        p.error("--inference-only and --eval-only are mutually exclusive")
    mode = "both"
    if args.inference_only:
        mode = "inference"
    elif args.eval_only:
        mode = "eval"

    inv = resolve_invocation(args, tasks, models)

    if inv["task"] == "all":
        selected = list(tasks.keys())
    else:
        selected = [inv["task"]]

    exit_code = 0
    check_env_model_compat(models, (inv["inference_model"], inv["judge_model"]))

    if args.slurm:
        return submit_slurm_jobs(args, inv, tasks, selected)

    for tname in selected:
        meta = tasks[tname]
        if not args.dry_run:
            preflight(meta)
        try:
            cmd = build_adapter_command(
                meta,
                inv["inference_model"],
                inv["judge_model"],
                inv["label"],
                inv["limit"],
                mode,
            )
        except Exception as e:
            sys.stderr.write(f"[{tname}] {e}\n")
            exit_code = 6
            continue

        print(f"[{tname}] " + " ".join(cmd))
        if args.dry_run:
            continue
        env = os.environ.copy()
        if "CREATIVITYPRISM_API_KEYS" not in env:
            api_keys_path = os.environ.get("CREATIVITYPRISM_API_KEYS", str(DEFAULT_API_KEYS))
            env["CREATIVITYPRISM_API_KEYS"] = api_keys_path
        if env.get("CREATIVITYPRISM_API_KEYS") and not os.path.isfile(env["CREATIVITYPRISM_API_KEYS"]):
            sys.stderr.write(
                f"Warning: CREATIVITYPRISM_API_KEYS={env['CREATIVITYPRISM_API_KEYS']} not found. "
                f"API-based models will fail.\n"
            )
        rc, stdout_lines = run_adapter(cmd, env)
        if rc != 0:
            sys.stderr.write(f"[{tname}] adapter exited with code {rc}\n")
            exit_code = rc

        target_dir, _metadata, warnings = artifacts.materialize_run_outputs(
            OUTPUTS_ROOT,
            inv["label"],
            meta,
            {**inv, "mode": mode},
            cmd,
            rc,
            stdout_lines,
        )
        for warning in warnings:
            sys.stderr.write(f"[{tname}] {warning}\n")
        print(f"[{tname}] centralized outputs: {target_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
