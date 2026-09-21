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
# The data files an artifact can consist of. A directory artifact is enumerated
# into an explicit list at link time so no later reader has to guess.
ARTIFACT_DATA_SUFFIXES = (".json", ".jsonl", ".csv")
MARKER_PREFIX = "CP_ARTIFACT"
LEGACY_MARKER_PREFIX = "OUTPUT_PATH="
METADATA_FILENAME = "metadata.json"
METADATA_SCHEMA_VERSION = 2


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


SIDECAR_FILENAME = ".cp_artifacts"


def read_sidecar_markers(target_dir):
    """Markers `emit_artifact` appended beside the run's outputs.

    Under SLURM the adapter's stdout is a batch log the runner never sees, so the
    sidecar is the only surviving copy. Missing or unreadable is not an error.
    """
    sidecar = Path(target_dir) / SIDECAR_FILENAME
    try:
        lines = sidecar.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}, []
    return parse_artifact_markers(lines)


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


def _artifact_files(native, is_dir):
    """The data files this artifact consists of, relative to `native`.

    A file artifact is itself. A directory artifact is every data file under it,
    sorted for stability. Non-data files (logs, checkpoints, .tmp) are left out:
    an artifact is what a reader should parse, not everything the task wrote.
    """
    if not is_dir:
        return [native.name]
    found = []
    for child in sorted(native.rglob("*")):
        if child.is_file() and child.suffix.lower() in ARTIFACT_DATA_SUFFIXES:
            found.append(str(child.relative_to(native)))
    return found


def _disambiguate_overlaps(records, warnings):
    """Remove files claimed by a narrower artifact from a broader one.

    A task that announces its whole output directory as `inference` announces a
    directory that also holds the evaluation files -- both kinds then legitimately
    point at the same bytes, and every reader downstream sees the evaluation rows
    twice, once mislabelled. The specific claim wins: a file named directly by one
    kind is subtracted from any other kind that merely contains it.
    """
    owned = {}
    for kind, record in records.items():
        if record.get("native_type") == "file" and record.get("exists"):
            owned[os.path.realpath(record["native_path"])] = kind

    for kind, record in records.items():
        if record.get("native_type") != "directory" or not record.get("exists"):
            continue
        root = Path(record["native_path"])
        keep, dropped = [], []
        for rel in record.get("files") or []:
            real = os.path.realpath(root / rel)
            other = owned.get(real)
            if other and other != kind:
                dropped.append({"file": rel, "claimed_by": other})
            else:
                keep.append(rel)
        if dropped:
            record["files"] = keep
            record["excluded"] = dropped
            warnings.append(
                "{} artifact is a directory that also contains the {} artifact; "
                "{} file(s) attributed to the more specific claim".format(
                    kind, ", ".join(sorted({d["claimed_by"] for d in dropped})),
                    len(dropped),
                )
            )


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
    # Enumerate now, while the run's own files are the only ones present.
    # Without this the loader had to glob the directory later and could not tell
    # a task's inference output from the evaluation files written beside it --
    # which is exactly how eval_output_cleaned.json got read back as inference.
    record["files"] = _artifact_files(native, is_dir)

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

    # stdout wins when both exist: it is the live run, while a stale sidecar can
    # survive from an earlier run into the same (label, task, model) directory.
    sidecar, sidecar_warnings = read_sidecar_markers(target_dir)
    if sidecar:
        warnings.extend(sidecar_warnings)
        announced = {**sidecar, **announced}

    records = {}
    for kind in ARTIFACT_KINDS:
        if kind in announced:
            records[kind] = link_artifact(
                target_dir, kind, announced[kind], base, warnings
            )

    _disambiguate_overlaps(records, warnings)

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
            # The env actually used, which CREATIVITYPRISM_FORCE_ENV can override.
            "environment": os.environ.get(
                "CREATIVITYPRISM_FORCE_ENV", task_meta.get("environment")
            ),
            "adapter": task_meta.get("adapter"),
            "command": list(command),
            "exit_code": exit_code,
        },
        records,
    )
    return target_dir, metadata, warnings
