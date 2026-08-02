#!/usr/bin/env python3
"""Adapter-to-runner artifact contract and centralized output materialization.

Adapters announce what they produced by printing marker lines on stdout:

    CP_ARTIFACT inference /abs/path/to/native/inference/output
    CP_ARTIFACT eval /abs/path/to/native/eval/output

The Phase 1 marker ``OUTPUT_PATH=<path>`` is still accepted and treated as an
``inference`` artifact, so adapters can migrate independently.

The runner never interprets artifact contents. It links each announced path
into ``outputs/{label}/{task}/{inference_model}/`` and records it in
``metadata.json``. Native task outputs are never copied or moved.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_KINDS = ("inference", "eval")
MARKER_PREFIX = "CP_ARTIFACT"
LEGACY_MARKER_PREFIX = "OUTPUT_PATH="
METADATA_FILENAME = "metadata.json"
METADATA_SCHEMA_VERSION = 1


def centralized_output_dir(outputs_root, label, task_name, inference_model):
    """The single canonical location for one (label, task, model) triple."""
    return Path(outputs_root) / label / task_name / inference_model


# ---------- Marker parsing ----------

def parse_artifact_markers(lines):
    """Extract artifact markers from adapter stdout.

    Returns ``(artifacts, warnings)`` where ``artifacts`` maps kind -> path
    string. Later markers of the same kind win. Performs no filesystem access.
    """
    artifacts = {}
    warnings = []
    legacy_path = None

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith(MARKER_PREFIX + " "):
            kind, _, path = line[len(MARKER_PREFIX):].strip().partition(" ")
            path = path.strip()
            if not path:
                warnings.append(f"malformed artifact marker (no path): {line}")
                continue
            if kind not in ARTIFACT_KINDS:
                warnings.append(
                    f"unknown artifact kind '{kind}'; expected one of "
                    + ", ".join(ARTIFACT_KINDS)
                )
                continue
            artifacts[kind] = path
        elif line.startswith(LEGACY_MARKER_PREFIX):
            path = line[len(LEGACY_MARKER_PREFIX):].strip()
            if path:
                legacy_path = path

    if legacy_path is not None and "inference" not in artifacts:
        artifacts["inference"] = legacy_path
    return artifacts, warnings


# ---------- Linking ----------

def _clear_managed_entries(target_dir, kind, warnings):
    """Remove links/references this module previously created for `kind`.

    Returns False if something we refuse to touch is in the way.
    """
    ok = True
    for entry in sorted(target_dir.iterdir()):
        if not entry.name.startswith(f"{kind}_output"):
            continue
        if entry.is_symlink():
            try:
                entry.unlink()
            except OSError:
                os.rmdir(entry)  # Windows directory symlinks need rmdir
        elif entry.is_dir():
            warnings.append(
                f"{entry} is a real directory, not a managed link; leaving it untouched"
            )
            ok = False
        else:
            entry.unlink()
    return ok


def _symlink_target(native, link_dir, base):
    """Relative target for in-repo artifacts so the tree survives relocation.

    Artifacts outside `base` get an absolute target; a relative one would be a
    long, brittle chain of parent hops.
    """
    try:
        resolved = native.resolve()
        if resolved.is_relative_to(base):
            return os.path.relpath(resolved, start=Path(link_dir).resolve())
    except (ValueError, OSError):  # different drive on Windows
        pass
    return str(native)


def link_artifact(target_dir, kind, native_path, base, warnings):
    """Link one native artifact into `target_dir`. Returns a metadata record."""
    native = Path(native_path)
    record = {"native_path": str(native), "exists": native.exists()}

    if not record["exists"]:
        warnings.append(
            f"adapter reported a {kind} artifact that does not exist: {native}"
        )
        record["link"] = None
        record["link_type"] = "missing"
        return record

    is_dir = native.is_dir()
    record["native_type"] = "directory" if is_dir else "file"

    if not _clear_managed_entries(target_dir, kind, warnings):
        record["link"] = None
        record["link_type"] = "blocked"
        return record

    link_name = f"{kind}_output" if is_dir else f"{kind}_output{native.suffix}"
    try:
        os.symlink(
            _symlink_target(native, target_dir, base),
            target_dir / link_name,
            target_is_directory=is_dir,
        )
        record["link"] = link_name
        record["link_type"] = "symlink"
    except (OSError, NotImplementedError) as exc:
        # Windows without Developer Mode, or filesystems without symlink support.
        ref_name = f"{kind}_output.path"
        (target_dir / ref_name).write_text(f"{native}\n", encoding="utf-8")
        record["link"] = ref_name
        record["link_type"] = "reference"
        record["link_error"] = str(exc)
        warnings.append(
            f"symlink unavailable for {kind} artifact ({exc}); "
            f"wrote path reference {ref_name} instead"
        )
    return record


# ---------- Metadata ----------

def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_metadata(target_dir, payload, artifacts):
    """Write metadata.json, merging artifacts from earlier runs of this triple."""
    path = target_dir / METADATA_FILENAME
    existing = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = {}

    merged = dict(existing.get("artifacts") or {})
    merged.update(artifacts)

    now = _utc_now()
    data = dict(payload)
    data.update({
        "schema_version": METADATA_SCHEMA_VERSION,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "artifacts": merged,
    })

    tmp = target_dir / (METADATA_FILENAME + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return data


# ---------- Orchestration ----------

def materialize_run_outputs(outputs_root, label, task_meta, invocation,
                            command, exit_code, stdout_lines):
    """Create the centralized view for one adapter run.

    Returns ``(target_dir, metadata, warnings)``.
    """
    announced, warnings = parse_artifact_markers(stdout_lines)
    task_name = task_meta["name"]
    outputs_root = Path(outputs_root)
    base = outputs_root.resolve().parent
    target_dir = centralized_output_dir(
        outputs_root, label, task_name, invocation["inference_model"]
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for kind in ARTIFACT_KINDS:
        if kind in announced:
            records[kind] = link_artifact(
                target_dir, kind, announced[kind], base, warnings
            )

    if not records:
        warnings.append(
            f"adapter for task '{task_name}' announced no artifacts "
            f"(expected a '{MARKER_PREFIX} <kind> <path>' line on stdout)"
        )

    metadata = write_metadata(
        target_dir,
        {
            "label": label,
            "task": task_name,
            "inference_model": invocation["inference_model"],
            "judge_model": invocation["judge_model"],
            "limit": invocation.get("limit"),
            "mode": invocation.get("mode"),
            "environment": task_meta.get("environment"),
            "adapter": task_meta.get("adapter"),
            "command": list(command),
            "exit_code": exit_code,
        },
        records,
    )
    return target_dir, metadata, warnings
