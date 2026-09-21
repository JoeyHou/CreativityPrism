#!/usr/bin/env python3
"""Assemble the Hugging Face dataset payload from the harness's own data.

    python release/build.py                 # -> release/creativityprism_v1
    python release/build.py --load-check    # also load every config with `datasets`

Layer A of the release is data only: prompts, the assets a task scores against,
the card, and the one script a user needs to complete the data. No harness code,
no judge prompt templates, no results. Building it from the harness's tree rather
than maintaining a second copy is what keeps the two from drifting -- the failure
mode is a prompt fixed in one place and not the other, which makes two scores
silently incomparable.

``data/`` mirrors ``tasks/prism_v1/data/processed`` exactly, and the files stay
JSON rather than becoming parquet: ``input`` and ``output`` have a different shape
in every task, so there is no one schema to flatten them into that does not throw
away what makes a task what it is.

A task whose registry entry says ``distribution: fetch`` has an unlicensed
upstream. Its rows are shipped with the upstream text set to null and a
``meta_data.source_ref`` pointing at the pinned upstream row; before blanking,
every row is checked against upstream, so a payload that ``fetch_upstream.py``
cannot turn back into the harness's data is never built.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tasks" / "prism_v1" / "data" / "processed"
CARD = REPO / "release" / "dataset_card.md"
USER_SCRIPTS = REPO / "release" / "scripts"
LICENSES = REPO / "release" / "licenses"
DEFAULT_OUT = REPO / "release" / "creativityprism_v1"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(USER_SCRIPTS))
from registry.read import load_tasks  # noqa: E402
import fetch_upstream  # noqa: E402  -- one download/checksum path, shared with users

# Not used to run anything: the TTCW paper's expert annotations and its own model
# generations. Shipped under research/ because judge-human agreement is checkable
# against them, but kept out of the task configs so nobody mistakes them for input.
RESEARCH = {
    "ttcw/ttcw_annotations.json",
    "ttcw/ttcw_annotations_full.json",
    "ttcw/claude.json",
    "ttcw/gpt3_5.json",
    "ttcw/gpt4.json",
}
# A three-item smoke file that duplicates rows from the real set.
SKIP = {"aut/aut_push_test.json"}


def fetch_entries():
    """(task, upstream, file entry) for every data file whose text we may not ship."""
    out = []
    for name, task in load_tasks(include_extra=False).items():
        if task.get("distribution", "bundled") != "fetch":
            continue
        upstream = task.get("upstream") or {}
        if not upstream.get("files"):
            raise SystemExit("{}: distribution is 'fetch' but it declares no "
                             "upstream.files, so there is nothing to fetch from".format(name))
        for entry in upstream["files"]:
            out.append((name, upstream, entry))
    return out


def blank(rows, upstream, entry):
    """Null out the upstream text, after proving fetch_upstream.py restores it exactly."""
    ref = {"url": upstream["url"], "commit": upstream["commit"],
           "path": entry["theirs"], "sha256": entry["sha256"]}
    theirs = fetch_upstream.download(ref)
    if len(rows) > len(theirs):
        raise SystemExit("{}: {} rows, but upstream {} has only {}".format(
            entry["ours"], len(rows), entry["theirs"], len(theirs)))
    prefix = entry.get("prefix", "")
    for i, row in enumerate(rows):
        expected = prefix + theirs[i][entry["field"]]
        current = fetch_upstream.get_path(row, entry["into"])
        # Already blank is fine (the harness copy may be purged later); anything
        # else must match, or the user would get different text than we scored.
        if current is not None and current != expected:
            raise SystemExit("{} row {} ({}): {} does not equal prefix + upstream {}[{}].{}"
                             .format(entry["ours"], i, row["meta_data"].get("id"),
                                     entry["into"], entry["theirs"], i, entry["field"]))
        fetch_upstream.set_path(row, entry["into"], None)
        row["meta_data"]["source_ref"] = dict(
            ref, index=i, field=entry["field"], into=entry["into"], prefix=prefix)
    return rows


def build(out_dir):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "research").mkdir()

    fetched = {e["ours"]: (up, e) for _, up, e in fetch_entries()}
    copied, research = [], []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(SOURCE))
        if rel in SKIP:
            continue
        if rel in RESEARCH:
            target = out / "research" / path.name
            research.append(rel)
        else:
            target = out / "data" / rel
            copied.append(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel in fetched:
            rows = blank(json.loads(path.read_text(encoding="utf-8")), *fetched[rel])
            fetch_upstream.write_atomic(target, rows)
        else:
            shutil.copy2(path, target)

    missing = set(fetched) - set(copied)
    if missing:
        raise SystemExit("registry names fetch files that are not in {}: {}".format(
            SOURCE, sorted(missing)))

    shutil.copytree(USER_SCRIPTS, out / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(LICENSES, out / "licenses")
    write_provenance(out)
    if CARD.is_file():
        shutil.copy2(CARD, out / "README.md")
    else:
        print("WARNING: no card at {}; the dataset will publish without one".format(CARD))
    return copied, research, sorted(fetched)


def write_provenance(out):
    """One machine-readable record of whose material each task is.

    Generated from the registry rather than written by hand: a provenance file
    that disagrees with what the harness runs is worse than none. Apache-2.0 and
    CC-BY upstreams require modifications to be stated, which is why every task
    carries `modifications`.
    """
    import yaml
    tasks = load_tasks(include_extra=False)
    entries = {}
    for name in sorted(tasks):
        task = tasks[name]
        up = task.get("upstream")
        if not up:
            continue
        entry = {k: up[k] for k in ("name", "url", "license", "paper", "modifications") if k in up}
        entry["distribution"] = task.get("distribution", "bundled")
        if (LICENSES / "{}.txt".format(name)).is_file():
            entry["license_file"] = "licenses/{}.txt".format(name)
        if entry["distribution"] == "fetch":
            entry["pinned_commit"] = up["commit"]
            entry["fetched_files"] = [f["theirs"] for f in up["files"]]
        entries[name] = entry
    missing = sorted(set(tasks) - set(entries) - {"contrib_metaphor"})
    if missing:
        raise SystemExit("no upstream declared for: {}. Every published task needs "
                         "one; add it to registry/tasks/<name>.yaml".format(missing))
    header = ("# Generated by release/build.py from registry/tasks/*.yaml.\n"
              "# Upstream material stays under its own licence; see licenses/ and the card.\n")
    (out / "provenance.yaml").write_text(
        header + yaml.safe_dump(entries, sort_keys=True, allow_unicode=True, width=88),
        encoding="utf-8")


def card_configs(out):
    import yaml
    front = (out / "README.md").read_text(encoding="utf-8").split("---")[1]
    return yaml.safe_load(front).get("configs") or []


def verify(out_dir, fetched):
    """What a user would trip over, checked before they do.

    Every config the card declares must match at least one file, and every file
    must be a JSON array. And no row of a fetch task may carry text: that is the
    one mistake here that cannot be taken back once it is pushed.
    """
    out = Path(out_dir)
    if not (out / "README.md").is_file():
        return ["no README.md"]
    problems = []
    for entry in card_configs(out):
        files = sorted(out.glob(entry["data_files"]))
        if not files:
            problems.append("{}: no file matches {}".format(
                entry["config_name"], entry["data_files"]))
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append("{}: {} is not valid JSON ({})".format(
                    entry["config_name"], f.name, exc))
                continue
            if not isinstance(data, list):
                problems.append("{}: {} is not a JSON array; the Hub's json loader "
                                "needs an array or jsonl".format(entry["config_name"], f.name))
    for rel in fetched:
        for i, row in enumerate(json.loads((out / "data" / rel).read_text(encoding="utf-8"))):
            ref = row["meta_data"].get("source_ref")
            if not ref:
                problems.append("{} row {}: no source_ref".format(rel, i))
            elif fetch_upstream.get_path(row, ref["into"]) is not None:
                problems.append("{} row {}: {} is not null -- upstream text would be "
                                "published".format(rel, i, ref["into"]))
    return problems


def load_check(out_dir):
    """Load every declared config the way a user would, with the real library."""
    from datasets import load_dataset
    out = Path(out_dir)
    for entry in card_configs(out):
        ds = load_dataset(str(out), entry["config_name"], split="train")
        print("  load_dataset {:24s} {:>6} rows".format(entry["config_name"], len(ds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--load-check", action="store_true",
                    help="also load every config with the `datasets` library")
    args = ap.parse_args()

    copied, research, fetched = build(args.out)
    print("data/     {} file(s), {} with upstream text removed".format(len(copied), len(fetched)))
    print("research/ {} file(s)".format(len(research)))
    print("licenses/ {} upstream licence(s), provenance.yaml written".format(
        len(list((Path(args.out) / "licenses").glob("*.txt")))))

    problems = verify(args.out, fetched)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("\nall declared configs present and parseable; no upstream text in fetch files")
    if args.load_check:
        load_check(args.out)
    print("-> {}".format(args.out))
    print("\nPush with:")
    print("  huggingface-cli upload joeyhou/creativityprism_v1 {} . --repo-type=dataset"
          .format(args.out))


if __name__ == "__main__":
    main()
