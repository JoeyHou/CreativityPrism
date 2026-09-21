"""Where the benchmark's data lives, and how it gets there.

    creativityprism data pull      # download it, then complete it
    creativityprism data status    # what is on disk, and at which dataset revision

This is the only module that knows the data comes from the Hub. Task code, the
adapters and the evaluation code are handed a directory and read files out of it;
none of them knows whether a file was shipped with the dataset or fetched from an
upstream repository afterwards.

Two steps, one command:

1. Download ``data/`` from the Hub dataset into the data root.
2. Run the dataset's own ``fetch_upstream.py``, which fills in the task text that
   the dataset may not redistribute -- currently Creativity Index, whose upstream
   carries no licence. Which tasks those are is declared in the registry, not
   here; the script works from what the rows themselves say.

Re-running it is a no-op once the data is there at the requested revision.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "joeyhou/creativityprism_v1"
BUNDLE = "prism_v1"
REVISION_FILE = ".dataset_revision"
# Shipped inside the dataset, so the fetch logic always matches the revision the
# data came from. A checkout has its own copy, which is the one it develops.
LOCAL_FETCH_SCRIPT = REPO_ROOT / "release" / "scripts" / "fetch_upstream.py"


def data_root():
    """The directory task code reads its inputs from.

    A checkout puts it where the harness has always kept it, so nothing about
    running from a clone changes. An installed copy has no business writing into
    site-packages, so it must be told where to put the data instead.
    """
    override = os.environ.get("CREATIVITYPRISM_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if (REPO_ROOT / "pyproject.toml").is_file():
        return REPO_ROOT / "tasks" / BUNDLE / "data" / "processed"
    raise SystemExit(
        "This is an installed copy of creativityprism, not a checkout, so there is\n"
        "no obvious place to keep the benchmark data. Choose one:\n"
        "    export CREATIVITYPRISM_DATA_DIR=/path/to/creativityprism-data\n"
        "then run `creativityprism data pull` again.")


def current_revision(root):
    f = Path(root) / REVISION_FILE
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("revision")
    except ValueError:
        return None


def _fetch_module(snapshot):
    """Prefer the checkout's own fetch script; fall back to the dataset's copy.

    Running the copy that came down from the Hub would mean executing code we
    just downloaded, which is worth avoiding when an identical file is sitting in
    the repository.
    """
    import importlib.util
    path = LOCAL_FETCH_SCRIPT if LOCAL_FETCH_SCRIPT.is_file() else Path(snapshot) / "scripts" / "fetch_upstream.py"
    if not path.is_file():
        raise SystemExit("no fetch_upstream.py in the checkout or in the downloaded "
                         "dataset; cannot complete the data")
    spec = importlib.util.spec_from_file_location("cp_fetch_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def pull(revision=None, root=None, force=False):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "`creativityprism data pull` needs huggingface_hub:\n"
            "    pip install huggingface_hub")

    root = Path(root) if root else data_root()
    root.mkdir(parents=True, exist_ok=True)

    snapshot = snapshot_download(REPO_ID, repo_type="dataset", revision=revision,
                                 allow_patterns=["data/*", "scripts/*"])
    # The directory name in the HF cache ends in the resolved commit.
    resolved = Path(snapshot).name
    have = current_revision(root)
    if have == resolved and not force and any(root.glob("*")):
        print("already at dataset revision {} in {}".format(have[:10], root))
        return root

    shutil.copytree(Path(snapshot) / "data", root, dirs_exist_ok=True)
    print("downloaded {} -> {}".format(REPO_ID, root))

    mod, script = _fetch_module(snapshot)
    print("completing the data with {}".format(script.name))
    _run_fetch(mod, root)

    (root / REVISION_FILE).write_text(json.dumps(
        {"repo_id": REPO_ID, "revision": resolved}, indent=2), encoding="utf-8")
    print("data root is at dataset revision {}".format(resolved[:10]))
    return root


def _run_fetch(mod, root):
    """Drive fetch_upstream.py's own argument parser, so there is one code path."""
    argv = sys.argv
    sys.argv = ["fetch_upstream.py", "--data-dir", str(root)]
    try:
        mod.main()
    finally:
        sys.argv = argv


def status(root=None):
    root = Path(root) if root else data_root()
    print("data root: {}".format(root))
    if not root.is_dir():
        print("  nothing here yet. Run: creativityprism data pull")
        return 1
    rev = current_revision(root)
    print("dataset:   {} @ {}".format(REPO_ID, rev[:10] if rev else "unknown revision"))

    sys.path.insert(0, str(REPO_ROOT))
    from registry import read as registry_read
    problems = 0
    for name, task in sorted(registry_read.load_tasks(include_extra=False).items()):
        up = task.get("upstream") or {}
        files = [e["ours"] for e in up.get("files", [])]
        if task.get("distribution", "bundled") != "fetch":
            continue
        for rel in files:
            path = root / rel
            if not path.is_file():
                print("  {:18s} {:40s} MISSING".format(name, rel))
                problems += 1
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            # A row without a source_ref came from a copy that already carries its
            # text -- a checkout that predates the split, or a task flipped back to
            # `bundled`. Only a row that points upstream can be waiting on a fetch.
            pending = [r for r in rows
                       if ((r.get("meta_data") or {}).get("source_ref") or {}).get("into")
                       and _get(r, r["meta_data"]["source_ref"]["into"]) is None]
            state = ("complete" if not pending
                     else "{} of {} rows still empty".format(len(pending), len(rows)))
            blank = len(pending)
            print("  {:18s} {:40s} {}".format(name, rel, state))
            problems += bool(blank)
    if problems:
        print("\nSome fetched data is missing. Run: creativityprism data pull")
    else:
        print("\nAll data present.")
    return 1 if problems else 0


def _get(row, dotted):
    node = row
    for key in (dotted or "").split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="creativityprism data",
        description="Download the benchmark data and complete it.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("pull", help="download the dataset, then fetch what it may not ship")
    pl.add_argument("--revision", help="dataset revision to pin (branch, tag or commit)")
    pl.add_argument("--data-dir", help="where to put it (default: $CREATIVITYPRISM_DATA_DIR "
                                       "or the checkout's tasks/{}/data/processed)".format(BUNDLE))
    pl.add_argument("--force", action="store_true", help="re-download and re-fetch anyway")
    st = sub.add_parser("status", help="what is on disk, and at which revision")
    st.add_argument("--data-dir")
    args = p.parse_args(argv)

    if args.cmd == "pull":
        pull(revision=args.revision, root=args.data_dir, force=args.force)
        return 0
    return status(root=args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
