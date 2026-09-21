"""One reader for registry/tasks/*.yaml, so nothing re-declares the task list.

Before this, the set of tasks was written out by hand in six places -- two
dispatch dicts, three gates, and the analysis loader -- and each copy could go
stale on its own. Two of today's bugs were exactly that: a list that no longer
matched what the code did, with nothing to notice. The YAML is the declaration;
everything else asks here.

No dependency on the runner, so a gate or a notebook can import it.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "registry" / "tasks"


def _extra_dirs():
    """Out-of-tree task directories, colon-separated.

    A contributor should be able to run their own task without forking this
    repo, which is what every comparable harness offers (lm-eval's
    --include_path, HELM's plug-in modules, lighteval's --custom-tasks).
    """
    raw = os.environ.get("CREATIVITYPRISM_INCLUDE_PATH", "")
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


def task_dirs():
    dirs = [TASKS_DIR] + _extra_dirs()
    return [d for d in dirs if d.is_dir()]


def load_tasks(include_extra=True):
    """name -> task dict, in registry order. Later directories win on a clash."""
    import yaml

    tasks = {}
    dirs = task_dirs() if include_extra else [TASKS_DIR]
    for directory in dirs:
        for f in sorted(Path(directory).glob("*.yaml")):
            if f.name.startswith("_"):
                continue
            with open(f) as fh:
                data = yaml.safe_load(fh) or {}
            name = data.get("name") or f.stem
            data.setdefault("name", name)
            data["_source"] = str(f)
            tasks[name] = data
    return tasks


def task_names(**kw):
    return sorted(load_tasks(**kw))


def adapters(**kw):
    """name -> absolute adapter path."""
    out = {}
    for name, data in load_tasks(**kw).items():
        adapter = data.get("adapter")
        if adapter:
            out[name] = (REPO_ROOT / adapter) if not os.path.isabs(adapter) else Path(adapter)
    return out


def judge_tasks(**kw):
    return {n for n, d in load_tasks(**kw).items() if d.get("uses_judge")}


def resolve_parser(spec, source=None):
    """Import a parser from '<module>:<callable>' or '<path.py>:<callable>'.

    Both spellings, because a contributed task lives in a directory that is not
    an importable package. Requiring the dotted form would have meant either
    adding an __init__.py to someone else's folder or putting their parser in
    our loader -- and the second is the central edit this whole arrangement
    exists to avoid. A relative path is resolved against the repo root, so a
    task can ship its parser beside its code and name it plainly:

        loader_parser: tasks/my_task/parser.py:parse
    """
    if not spec or ":" not in spec:
        return None
    target, _, attr = spec.rpartition(":")
    if not target:
        return None

    import importlib
    import importlib.util
    import sys

    if target.endswith(".py") or os.sep in target or "/" in target:
        path = Path(target)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            return None
        module_name = "cp_task_parser_" + path.stem
        try:
            spec_obj = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(module)
        except Exception:                                   # noqa: BLE001
            return None
        return getattr(module, attr, None)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        module = importlib.import_module(target)
    except ImportError:
        return None
    return getattr(module, attr, None)


def parsers(**kw):
    """name -> parser callable, for tasks that declare a resolvable one."""
    out = {}
    for name, data in load_tasks(**kw).items():
        fn = resolve_parser(data.get("loader_parser"))
        if fn is not None:
            out[name] = fn
    return out


DEFAULT_STAGES = ("inference", "eval")


def stages(name, **kw):
    """The stages a task declares, defaulting to both."""
    data = load_tasks(**kw).get(name) or {}
    declared = data.get("stages")
    return tuple(declared) if declared else DEFAULT_STAGES


BUILTIN_FOLDER = "tasks/prism_v1"


def builtin_tasks(**kw):
    """Tasks implemented inside the built-in bundle.

    The bundle has internal conventions -- one run_id variable, a fixed
    inference_output.json, two dispatch dicts -- that are worth asserting *for
    it* and wrong to impose on anyone else. A contributed task keeps its own
    layout; that is the whole point of it being contributed.
    """
    return {n for n, d in load_tasks(**kw).items()
            if (d.get("folder") or "").rstrip("/") == BUILTIN_FOLDER}


def contributed_tasks(**kw):
    return set(load_tasks(**kw)) - builtin_tasks(**kw)
