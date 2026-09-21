#!/usr/bin/env python3
"""Fill in the task text this dataset is not allowed to ship.

    python scripts/fetch_upstream.py                 # fills data/ next to this script
    python scripts/fetch_upstream.py --data-dir DIR  # or any copy of that data/

Some tasks come from upstream repositories that carry no license, so their text
cannot be redistributed. For those, every row is still here -- same ids, same
order, same fields -- but the text is null, and ``meta_data.source_ref`` says
exactly where it lives upstream: repository, pinned commit, file, row index, and
the file's sha256. This script downloads each referenced file from that commit,
refuses to continue if its checksum differs, and writes the text into the rows
on your machine. Use of that text is governed by the upstream authors' terms.

Standard library only. Re-running it is a no-op once the data is filled in.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ISSUES = "https://github.com/JoeyHou/CreativityPrism/issues"


def raw_url(repo_url, commit, path):
    owner_repo = repo_url.rstrip("/").split("github.com/", 1)[1]
    return "https://raw.githubusercontent.com/{}/{}/{}".format(owner_repo, commit, path)


def get_path(row, dotted):
    node = row
    for key in dotted.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def set_path(row, dotted, value):
    keys = dotted.split(".")
    node = row
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def fetch_files(data_dir):
    """Every data file with at least one row that points upstream."""
    found = []
    for path in sorted(Path(data_dir).rglob("*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(rows, list) and any(
                isinstance(r, dict) and (r.get("meta_data") or {}).get("source_ref")
                for r in rows):
            found.append((path, rows))
    return found


def download(ref):
    url = raw_url(ref["url"], ref["commit"], ref["path"])
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "creativityprism-fetch"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        sys.exit("Could not download {}\n  ({})\nCheck your network connection and "
                 "try again. The upstream repository is {}".format(url, exc, ref["url"]))
    digest = hashlib.sha256(body).hexdigest()
    if digest != ref["sha256"]:
        sys.exit(
            "CHECKSUM MISMATCH for {}\n"
            "  expected sha256 {}\n"
            "  got             {}\n"
            "The upstream file is no longer the one this dataset was built against, so\n"
            "scores computed from it would not be comparable to anyone else's. Stopping\n"
            "before writing the rows that need it. Please open an issue at {}".format(
                url, ref["sha256"], digest, ISSUES))
    return json.loads(body.decode("utf-8"))


def write_atomic(path, rows):
    # indent=4 and no trailing newline, as the harness writes its own files. A
    # filled file then differs from the harness's copy only by the source_ref keys.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rows, indent=4))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"),
                    help="the dataset's data/ directory (default: the one beside this script)")
    ap.add_argument("--force", action="store_true",
                    help="download and rewrite even rows that are already filled in")
    args = ap.parse_args()

    if not Path(args.data_dir).is_dir():
        sys.exit("No data directory at {}. Point --data-dir at the dataset's data/ "
                 "folder.".format(args.data_dir))

    files = fetch_files(args.data_dir)
    if not files:
        print("Nothing to fetch under {}: no row points upstream.".format(args.data_dir))
        return

    cache = {}
    for path, rows in files:
        refs = [(r.get("meta_data") or {}).get("source_ref") for r in rows]
        todo = [i for i, ref in enumerate(refs)
                if ref and (args.force or get_path(rows[i], ref["into"]) is None)]
        rel = path.relative_to(args.data_dir)
        if not todo:
            print("ok       {}  ({} rows already filled in)".format(rel, len(rows)))
            continue
        for i in todo:
            ref = refs[i]
            key = (ref["url"], ref["commit"], ref["path"])
            if key not in cache:
                cache[key] = download(ref)
            upstream = cache[key]
            value = upstream[ref["index"]][ref["field"]]
            set_path(rows[i], ref["into"], ref.get("prefix", "") + value)
        write_atomic(path, rows)
        print("fetched  {}  ({} rows from {} @ {})".format(
            rel, len(todo), refs[todo[0]]["url"], refs[todo[0]]["commit"][:10]))


if __name__ == "__main__":
    main()
