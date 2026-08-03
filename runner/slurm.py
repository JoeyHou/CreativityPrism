#!/usr/bin/env python3
"""SLURM submission for the unified runner.

The generated sbatch script re-invokes ``runner/run.py`` with the same arguments
minus ``--slurm``. The runner inside the job therefore performs its usual artifact
materialization, so nothing about the local path has to be duplicated here.

Directive values come from ``runner/slurm_template.sbatch`` defaults, overlaid with
the task's ``slurm:`` block in ``registry/tasks/{name}.yaml``, overlaid with any
``--slurm-override key=value`` flags. A directive resolving to an empty value is
omitted, which is how a CPU-only run drops ``--gres``.
"""
import re
import shlex
import subprocess
from pathlib import Path

TEMPLATE_NAME = "slurm_template.sbatch"
SCRIPTS_DIRNAME = "slurm_scripts"

# Directives the template understands, in emission order.
DIRECTIVE_ORDER = (
    "job-name", "output", "error", "partition", "account",
    "time", "mem", "cpus-per-task", "gres", "nodes", "ntasks",
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value):
    """Filesystem- and SLURM-safe fragment for job names and script paths."""
    return _SAFE_NAME.sub("_", str(value)).strip("_") or "run"


def load_template_defaults(template_path):
    """Parse ``#SBATCH --key=value`` defaults out of the template."""
    defaults = {}
    for line in Path(template_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH "):
            continue
        body = line[len("#SBATCH "):].strip()
        if not body.startswith("--"):
            continue
        key, _, value = body[2:].partition("=")
        defaults[key.strip()] = value.strip()
    return defaults


def parse_overrides(pairs):
    """``["time=8:00:00", "gres="]`` -> ``{"time": "8:00:00", "gres": ""}``."""
    overrides = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--slurm-override expects key=value, got '{pair}'")
        key = key.strip().lstrip("-")
        if not key:
            raise ValueError(f"--slurm-override has an empty key: '{pair}'")
        overrides[key] = value.strip()
    return overrides


def resolve_directives(template_defaults, task_meta, overrides, job_name, log_dir):
    """Merge the three layers and drop anything that resolved to empty."""
    merged = dict(template_defaults)
    for key, value in (task_meta.get("slurm") or {}).items():
        merged[str(key).replace("_", "-")] = "" if value is None else str(value)
    merged.update(overrides)

    merged.setdefault("job-name", job_name)
    if not merged.get("job-name"):
        merged["job-name"] = job_name
    for stream, suffix in (("output", "out"), ("error", "err")):
        if not merged.get(stream):
            merged[stream] = f"{log_dir}/%x-%j.{suffix}"

    ordered = [(k, merged[k]) for k in DIRECTIVE_ORDER if merged.get(k)]
    ordered += [(k, v) for k, v in sorted(merged.items())
                if v and k not in DIRECTIVE_ORDER]
    return ordered


def strip_slurm_flags(argv):
    """Remove --slurm/--no-submit/--slurm-override from a runner argv."""
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--slurm", "--no-submit"):
            continue
        if arg == "--slurm-override":
            skip_next = True
            continue
        if arg.startswith("--slurm-override="):
            continue
        cleaned.append(arg)
    return cleaned


def render_script(template_path, directives, inner_command):
    body = Path(template_path).read_text(encoding="utf-8")
    header = "\n".join(f"#SBATCH --{key}={value}" for key, value in directives)
    # Template directives are only defaults; the resolved header replaces them
    # wholesale. #~ lines document the template and never reach the job script.
    body = "\n".join(
        line for line in body.splitlines()
        if not line.strip().startswith(("#SBATCH ", "#~"))
    )
    body = body.replace("{{SBATCH_DIRECTIVES}}", header)
    body = body.replace("{{COMMAND}}", inner_command)
    if not body.endswith("\n"):
        body += "\n"
    return body


def build_job(repo_root, task_name, task_meta, inference_model, label,
              runner_argv, overrides, python_executable="python3"):
    """Render one job script and return ``(script_path, content)``.

    ``python_executable`` deliberately defaults to bare ``python3`` rather than
    ``sys.executable``: the script may be generated on a laptop and run on the
    cluster, where the local interpreter path does not exist.
    """
    template_path = Path(repo_root) / "runner" / TEMPLATE_NAME
    if not template_path.is_file():
        raise FileNotFoundError(f"SLURM template not found: {template_path}")

    job_name = f"cp-{safe_name(label)}-{safe_name(task_name)}-{safe_name(inference_model)}"
    script_dir = Path(repo_root) / SCRIPTS_DIRNAME / safe_name(label)
    log_dir = f"{SCRIPTS_DIRNAME}/{safe_name(label)}/logs"
    (script_dir / "logs").mkdir(parents=True, exist_ok=True)

    directives = resolve_directives(
        load_template_defaults(template_path), task_meta, overrides, job_name, log_dir
    )

    inner_argv = [python_executable, "runner/run.py"]
    inner_argv += strip_slurm_flags(runner_argv)
    # The job runs exactly one task even when the caller said --task all, so each
    # task gets its own queue slot and its own resource request.
    inner_argv = _force_single_task(inner_argv, task_name)
    inner_command = " ".join(shlex.quote(part) for part in inner_argv)

    content = render_script(template_path, directives, inner_command)
    script_path = script_dir / f"{safe_name(task_name)}_{safe_name(inference_model)}.sbatch"
    script_path.write_text(content, encoding="utf-8")
    return script_path, content


def _force_single_task(argv, task_name):
    out = []
    replaced = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--task" and index + 1 < len(argv):
            out += ["--task", task_name]
            replaced = True
            index += 2
            continue
        if arg.startswith("--task="):
            out.append(f"--task={task_name}")
            replaced = True
            index += 1
            continue
        out.append(arg)
        index += 1
    if not replaced:
        out += ["--task", task_name]
    return out


def submit(script_path):
    """``sbatch`` the script. Returns ``(job_id, message)``; job_id is None on failure."""
    try:
        proc = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None, "sbatch not found on PATH; use --no-submit to only generate scripts"
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "sbatch failed").strip()
    out = (proc.stdout or "").strip()
    match = re.search(r"(\d+)", out)
    return (match.group(1) if match else None), out
