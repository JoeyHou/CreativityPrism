#!/usr/bin/env python3
"""Load runner outputs into one table.

    from result_analysis.loader import load_outputs   # cwd = repo root
    import loader                                     # cwd = result_analysis/

    df = load_outputs(run_id="v3")
    df = load_outputs(run_id="v3", task="aut", model="GPT4.1")

Both spellings work; which one you need depends on the working directory, and a
notebook sitting in ``result_analysis/`` has to use the second. There is no
``__init__.py`` -- the dotted form resolves as a namespace package.

``run_id`` is the ``--label`` the run was launched with, which is the directory name
under ``outputs/``.

Only ``outputs/`` is read. Native artifact paths come from each run's
``metadata.json``, never from guessing where a task happens to write, so the loader
does not have to know any task's directory layout -- only its file format.

pandas is an optional dependency: ``load_records`` returns plain dicts and needs
nothing, ``load_outputs`` wraps them in a DataFrame. The gate can therefore run in an
interpreter that has no pandas installed.

Two conventions worth knowing before reading the numbers:

**The loader flattens, it does not aggregate.** One row per scored unit, never a mean.
`aut` scores every extracted use case, `creativity_index` scores every n-gram size,
`neocoder` scores every denial round, and `creative_math` returns three separate
verdicts per problem. Which of those to average, and how, is an analysis decision that
belongs in the notebook, not in the loader.

**`eval_score` is only meaningful next to `metric`.** A DAT distance of 86, an n-gram
coverage of 0.45 and a YES/NO judge verdict are not comparable quantities, so every row
names the metric its score belongs to. Filter on `metric` before aggregating.
"""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"
METADATA_FILENAME = "metadata.json"

COLUMNS = ("run_id", "task", "model", "sample_id", "metric", "prompt", "output", "eval_score")


class LoaderWarning(UserWarning):
    pass


# ---------- Artifact resolution ----------

def _reroot(native_path):
    """Re-root an absolute path recorded on another machine under this repo.

    ``metadata.json`` stores absolute paths, so a run performed on the cluster and
    analysed on a laptop (or vice versa) would otherwise be unreadable. Every task
    writes inside ``tasks/``, which gives a stable anchor.
    """
    parts = Path(native_path.replace("\\", "/")).parts
    if "tasks" not in parts:
        return None
    candidate = REPO_ROOT.joinpath(*parts[parts.index("tasks"):])
    return candidate if candidate.exists() else None


def resolve_artifact(run_dir, record):
    """Absolute path to one artifact, or None.

    Tries, in order: the symlink the runner created, the ``.path`` reference file it
    falls back to on Windows, the recorded native path, and finally that path
    re-rooted under this repo.
    """
    if not record:
        return None
    run_dir = Path(run_dir)

    link = record.get("link")
    if link:
        link_path = run_dir / link
        if link.endswith(".path"):
            if link_path.is_file():
                target = Path(link_path.read_text(encoding="utf-8").strip())
                if target.exists():
                    return target
        elif link_path.exists():
            return link_path.resolve()

    native = record.get("native_path")
    if native:
        native_path = Path(native)
        if native_path.exists():
            return native_path
        return _reroot(native)
    return None


def iter_run_dirs(run_id, task=None, model=None, outputs_root=None):
    """Yield ``(metadata, run_dir)`` for every model directory in a run."""
    root = Path(outputs_root or OUTPUTS_ROOT) / run_id
    if not root.is_dir():
        raise FileNotFoundError(
            f"no such run: {root}. Available: "
            f"{sorted(p.name for p in Path(outputs_root or OUTPUTS_ROOT).glob('*') if p.is_dir())}"
        )
    for meta_path in sorted(root.glob(f"*/*/{METADATA_FILENAME}")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task and meta.get("task") != task:
            continue
        if model and meta.get("inference_model") != model:
            continue
        yield meta, meta_path.parent


def _read_json(path):
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path):
    """A CSV as ``[{column: text}, ...]``, so a parser can treat it like JSON rows.

    Values stay strings; `_as_number` already turns "False"/"0.75" into a float,
    and guessing types here would only move that decision somewhere less visible.
    """
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def load_artifact(path):
    """Read one artifact as ``[(name, data), ...]``.

    An artifact is a file for some tasks and a directory for others -- `dat` writes
    inference into a directory but evaluation into a file, and `creativity_index`
    writes one evaluation file per n-gram size. Normalising both shapes here keeps
    that variation out of the per-task parsers.

    CSV is read as well as JSON because `neocoder`'s announced evaluation artifact
    is a CSV. Reading it, rather than the JSON sitting in a sibling directory the
    task never announced, keeps the loader honest to the CP_ARTIFACT contract.
    """
    if path is None:
        return []
    path = Path(path)
    if path.is_dir():
        out = []
        for child in sorted(path.rglob("*.json")) + sorted(path.rglob("*.csv")):
            data = _read_json(child) if child.suffix == ".json" else _read_csv(child)
            if data is not None:
                out.append((child.stem, data))
        return out
    data = _read_json(path) if path.suffix == ".json" else _read_csv(path)
    return [(path.stem, data)] if data is not None else []


# ---------- Per-task parsers ----------
# Each takes the loaded [(name, data), ...] pairs for the inference and evaluation
# artifacts and returns row dicts with sample_id/metric/prompt/output/eval_score;
# run_id/task/model are added by the caller. A parser must tolerate a missing
# evaluation, since --inference-only is a valid run.

def _first(mapping, *keys):
    for key in keys:
        if isinstance(mapping, dict) and mapping.get(key) not in (None, ""):
            return mapping[key]
    return None


def _as_number(value):
    """Numeric score, or None. Booleans and YES/NO verdicts become 1.0/0.0."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().upper()
        if text in ("YES", "TRUE", "CORRECT"):
            return 1.0
        if text in ("NO", "FALSE", "INCORRECT"):
            return 0.0
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _only_list(pairs):
    """The first artifact in ``pairs`` whose payload is a list."""
    for _name, data in pairs:
        if isinstance(data, list):
            return data
    return []


def _parse_dat(inference, evaluation):
    """`dat`: inference `{problem_statement, output}`, evaluation `{words, score}`.

    Index-aligned; there is no id field to join on.
    """
    scores = _only_list(evaluation)
    rows = []
    for _name, items in inference:
        for index, item in enumerate(items or []):
            escore = None
            if index < len(scores) and isinstance(scores[index], dict):
                escore = _as_number(scores[index].get("score"))
            rows.append({
                "sample_id": index,
                "metric": "dat_score" if escore is not None else None,
                "prompt": _first(item, "problem_statement", "prompt", "input"),
                "output": _first(item, "output", "response"),
                "eval_score": escore,
            })
    return rows


def _pick(pairs, *names):
    """The artifact whose file stem matches one of ``names``.

    `aut`, `ttcw` and `creative_short` announce a whole directory as their inference
    artifact, and that directory also holds the evaluation files. Selecting by name
    stops the eval output from being read back in as inference.
    """
    for name in names:
        for stem, data in pairs:
            if stem == name:
                return data
    return None


def _parse_aut(inference, evaluation):
    """`aut`: one row per use case the judge scored.

    Inference is `{sample_id: {prompt_variant: response}}`. Evaluation is a list whose
    `prompt_id` is `[sample_id, prompt_variant]` and whose `cleaned_output` holds
    `[use_case_text, novelty_score]` pairs -- one per use case, since a single
    response proposes several uses and each is scored 1-5 separately.

    `cleaned_output` is empty when the judge's reply contained no parsable score, so
    a sample is still emitted unscored rather than silently dropped.
    """
    responses = {}
    data = _pick(inference, "inference_output")
    if isinstance(data, dict):
        for sid, variants in data.items():
            if isinstance(variants, dict):
                for variant, text in variants.items():
                    responses[(str(sid), str(variant))] = text

    rows = []
    scored = set()
    for entry in _pick(evaluation, "eval_output_cleaned") or []:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("prompt_id") or []
        sid, variant = (list(pid) + [None, None])[:2]
        pairs = entry.get("cleaned_output") or []
        if not pairs:
            continue
        scored.add((str(sid), str(variant)))
        for index, pair in enumerate(pairs):
            text, score = (list(pair) + [None, None])[:2] \
                if isinstance(pair, (list, tuple)) else (pair, None)
            rows.append({
                "sample_id": f"{sid}|{variant}|{index}",
                "metric": "novelty",
                # Deliberately None, not the entry's `prompt_text`: for this task that
                # field holds the judge's rubric, not the prompt the model under test
                # answered. The object prompt is not in either artifact -- only the
                # sample id is -- so it stays empty rather than quietly wrong.
                "prompt": None,
                "output": text,
                "eval_score": _as_number(score),
            })

    for (sid, variant), text in sorted(responses.items()):
        if (sid, variant) not in scored:
            rows.append({
                "sample_id": f"{sid}|{variant}",
                "metric": None,
                "prompt": None,
                "output": text,
                "eval_score": None,
            })
    return rows


def _parse_ttcw(inference, evaluation):
    """`ttcw`: one row per (story, rubric question).

    Each story is put to a fixed set of binary rubric questions, and `cleaned_output`
    is that question's 0/1 verdict. The question id is the second element of
    `prompt_id`, so it becomes the metric rather than part of the sample id -- the
    unit of analysis is the story.
    """
    stories = {}
    data = _pick(inference, "inference_output")
    if isinstance(data, dict):
        for sid, record in data.items():
            if isinstance(record, dict):
                stories[str(sid)] = record

    rows = []
    scored = set()
    for entry in _pick(evaluation, "eval_output_cleaned") or []:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("prompt_id") or []
        sid, question = (list(pid) + [None, None])[:2]
        scored.add(str(sid))
        source = stories.get(str(sid), {})
        rows.append({
            "sample_id": str(sid),
            "metric": f"rubric_q{question}",
            "prompt": source.get("prompt_text"),
            "output": source.get("raw_output"),
            "eval_score": _as_number(entry.get("cleaned_output")),
        })

    for sid, record in sorted(stories.items()):
        if sid not in scored:
            rows.append({
                "sample_id": sid,
                "metric": None,
                "prompt": record.get("prompt_text"),
                "output": record.get("raw_output"),
                "eval_score": None,
            })
    return rows


def _parse_creative_short(inference, evaluation):
    """`creative_short`: one row per (story, automatic metric).

    Evaluation is keyed by sample id and carries `eval_result` with several unrelated
    numbers -- `dsi`, `surprise`, and `n_gram_diversity` for n = 1..5. They are
    emitted as separate metrics because they are different measurements, not repeats.
    """
    stories = _pick(inference, "inference_output") or {}
    scored = _pick(evaluation, "eval_output_cleaned") or {}

    rows = []
    for sid, record in sorted(stories.items()) if isinstance(stories, dict) else []:
        if not isinstance(record, dict):
            continue
        prompt = record.get("prompt_text")
        output = _first(record, "cleaned_output", "raw_output")
        result = (scored.get(sid) or {}).get("eval_result") \
            if isinstance(scored, dict) else None
        if not isinstance(result, dict):
            rows.append({"sample_id": sid, "metric": None, "prompt": prompt,
                         "output": output, "eval_score": None})
            continue
        for name, value in result.items():
            if isinstance(value, list):
                for order, item in enumerate(value, start=1):
                    rows.append({"sample_id": sid, "metric": f"{name}_{order}",
                                 "prompt": prompt, "output": output,
                                 "eval_score": _as_number(item)})
            else:
                rows.append({"sample_id": sid, "metric": name, "prompt": prompt,
                             "output": output, "eval_score": _as_number(value)})
    return rows


def _parse_ttct(inference, evaluation):
    """`ttct`: one row per (question, prompt variant).

    Each item carries three prompt variants (`text_basic`, `text_instructive`,
    `text_cot`) and only `text_cot` is scored; the rest are stored as "SKIPPED".

    The evaluation file drops unscored items **and carries no id**, so it cannot be
    joined by position or by key. It is joined on `question_type` plus the exact
    `text_cot` prompt, which is the only stable identifier both files share.

    The judge answers in prose, so `eval_score` is filled only when that answer is
    itself a number. Regex-mining a score out of free text would be worse than a null.
    """
    scored = {}
    for _name, data in evaluation:
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict):
                key = (item.get("question_type"), (item.get("input") or {}).get("text_cot"))
                scored[key] = item.get("evaluation") or {}

    rows = []
    for _name, data in inference:
        for index, item in enumerate(data if isinstance(data, list) else []):
            if not isinstance(item, dict) or item.get("skip"):
                continue
            meta = item.get("meta_data") or {}
            inputs = item.get("input") or {}
            outputs = item.get("output") or {}
            verdicts = scored.get((meta.get("question_type"), inputs.get("text_cot")), {})
            for variant, response in outputs.items():
                verdict = verdicts.get(variant)
                if isinstance(verdict, str) and verdict.strip().upper() == "SKIPPED":
                    verdict = None
                rows.append({
                    "sample_id": f"{meta.get('id')}|{index}|{variant}",
                    "metric": "judge_score" if verdict is not None else None,
                    "prompt": inputs.get(variant),
                    "output": response,
                    "eval_score": _as_number(verdict),
                })
    return rows


CREATIVE_MATH_CRITERIA = ("correctness", "coarse_grained_novelty", "fine_grained_novelty")


def _parse_creative_math(inference, evaluation):
    """`creative_math`: three independent YES/NO verdicts per problem.

    Each verdict is a majority vote of a fixed three-judge panel. All three are
    emitted as separate rows, because `correctness` and the two novelty grades
    answer different questions and averaging them together is meaningless.
    """
    scored = {}
    for _name, data in evaluation:
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict):
                scored[(str(item.get("problem_id")), item.get("question_number"))] = item

    rows = []
    for _name, data in inference:
        for index, item in enumerate(data if isinstance(data, list) else []):
            if not isinstance(item, dict):
                continue
            pid = str(item.get("problem_id"))
            qnum = item.get("question_number")
            key = f"{pid}|{qnum}"
            prompt = item.get("problem")
            output = _first(item, "cleaned_response", "response")
            verdicts = scored.get((pid, qnum))
            if not verdicts:
                rows.append({"sample_id": key, "metric": None, "prompt": prompt,
                             "output": output, "eval_score": None})
                continue
            for criterion in CREATIVE_MATH_CRITERIA:
                block = verdicts.get(criterion)
                # The panel is fixed, so all three criteria were always attempted.
                # An absent block and an empty one both mean "no decision came
                # back", and both stay as an unscored row rather than vanishing.
                decision = block.get("final_decision") if isinstance(block, dict) else None
                rows.append({
                    "sample_id": key,
                    "metric": criterion,
                    "prompt": prompt,
                    "output": output,
                    "eval_score": _as_number(decision),
                })
    return rows


def _parse_creativity_index(inference, evaluation):
    """`creativity_index`: one row per (document, n-gram size).

    Each evaluation file copies the whole inference record and adds `coverage`, so
    no join is needed. The n-gram size lives only in the file name
    (`{alias}_{domain}_exact_{min_ngram}.json`) and is recovered from it; coverage
    falls monotonically as the size grows, which is why every size is kept as its
    own metric instead of being collapsed into one number.

    Note that `dataset` is the literal string "creativity_index" and not the domain.
    The domain is carried by the `prompt_id` prefix (`book_0`) and by the file name.
    """
    rows = []
    scored = set()
    for name, data in evaluation:
        parts = name.split("_")
        ngram = parts[parts.index("exact") + 1] if "exact" in parts[:-1] else None
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("prompt_id") or item.get("doc_id"))
            scored.add(doc_id)
            rows.append({
                "sample_id": doc_id,
                "metric": f"coverage_exact_{ngram}",
                "prompt": item.get("prompt"),
                "output": item.get("response"),
                "eval_score": _as_number(item.get("coverage")),
            })

    # --inference-only, or a domain the eval sweep did not reach.
    for name, data in inference:
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("prompt_id") or f"{name}_{len(rows)}")
            if doc_id in scored:
                continue
            rows.append({
                "sample_id": doc_id,
                "metric": None,
                "prompt": item.get("prompt"),
                "output": item.get("response"),
                "eval_score": None,
            })
    return rows


NEOCODER_METRICS = ("correctness", "follow_constraints", "new_techniques_ratio")


def _parse_neocoder(inference, evaluation):
    """`neocoder`: one row per (problem, denial round, metric).

    Round 0 is the unconstrained problem and each later round forbids more
    techniques, so the rounds are a difficulty ladder rather than repeats. The
    prompt and output text live in the inference artifact as parallel lists; the
    scores live in the announced evaluation artifact, which is a CSV with one row
    per (problem_id, dp). The CSV is shorter than the inference -- evaluation drops
    rounds it could not score -- so the join is on the key, never on position.
    """
    prompts, outputs = {}, {}
    for _name, data in inference:
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("problem_id"))
            statements = item.get("problem_statements") or []
            produced = item.get("outputs") or []
            for index in range(max(len(statements), len(produced))):
                if index < len(statements):
                    prompts[(pid, index)] = statements[index]
                if index < len(produced):
                    outputs[(pid, index)] = produced[index]

    rows, scored = [], set()
    for _name, data in evaluation:
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict) or "dp" not in entry:
                continue
            pid = str(entry.get("problem_id"))
            try:
                index = int(entry["dp"])
            except (TypeError, ValueError):
                continue
            key = (pid, index)
            scored.add(key)
            for metric in NEOCODER_METRICS:
                rows.append({
                    "sample_id": f"{pid}|{index}",
                    "metric": metric,
                    "prompt": prompts.get(key),
                    "output": outputs.get(key),
                    "eval_score": _as_number(entry.get(metric)),
                })

    # A round the evaluation dropped still ran, so it stays as an unscored row.
    # Silently discarding it would make the generation count look like the
    # evaluation count and hide how much the evaluator skipped.
    for key in sorted(set(prompts) | set(outputs)):
        if key in scored:
            continue
        rows.append({
            "sample_id": f"{key[0]}|{key[1]}",
            "metric": None,
            "prompt": prompts.get(key),
            "output": outputs.get(key),
            "eval_score": None,
        })
    return rows


PARSERS = {
    "aut": _parse_aut,
    "ttcw": _parse_ttcw,
    "creative_short": _parse_creative_short,
    "ttct": _parse_ttct,
    "creative_math": _parse_creative_math,
    "creativity_index": _parse_creativity_index,
    "neocoder": _parse_neocoder,
    "dat": _parse_dat,
}


# ---------- Public API ----------

def load_records(run_id, task=None, model=None, outputs_root=None, strict=False):
    """Rows for one run as plain dicts. No pandas required."""
    rows = []
    problems = []
    for meta, run_dir in iter_run_dirs(run_id, task, model, outputs_root):
        task_name = meta.get("task")
        parser = PARSERS.get(task_name)
        if parser is None:
            problems.append(f"{task_name}: no parser registered")
            continue

        artifacts = meta.get("artifacts") or {}
        inference = load_artifact(resolve_artifact(run_dir, artifacts.get("inference")))
        evaluation = load_artifact(resolve_artifact(run_dir, artifacts.get("eval")))
        if not inference:
            problems.append(f"{task_name}: no readable inference artifact under {run_dir}")
            continue

        for row in parser(inference, evaluation):
            row.update({
                "run_id": run_id,
                "task": task_name,
                "model": meta.get("inference_model"),
            })
            rows.append({key: row.get(key) for key in COLUMNS})

    if problems:
        message = "; ".join(problems)
        if strict:
            raise ValueError(message)
        import warnings
        warnings.warn(message, LoaderWarning, stacklevel=2)
    return rows


def load_outputs(run_id, task=None, model=None, outputs_root=None, strict=False):
    """Rows for one run as a pandas DataFrame with :data:`COLUMNS`."""
    import pandas as pd
    return pd.DataFrame(
        load_records(run_id, task, model, outputs_root, strict),
        columns=list(COLUMNS),
    )


def list_runs(outputs_root=None):
    root = Path(outputs_root or OUTPUTS_ROOT)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
