"""Turn this task's artifacts into rows for the shared analysis table.

Named by registry/tasks/contrib_metaphor.yaml (`loader_parser`). It lives here,
beside the task, rather than in result_analysis/loader.py -- adding a task
should not mean editing a central file.

Contract: takes (inference, evaluation), each a list of ``(file_stem, data)``
pairs, and returns row dicts with sample_id / metric / prompt / output /
eval_score. run_id, task and model are filled in by the caller.
"""


def parse(inference, evaluation):
    # Inference is a CSV read as a list of dicts; evaluation is JSON. The task
    # gives its rows no id at all, so position is the only join -- recorded in
    # sample_id so the ambiguity is visible rather than implied.
    gen_rows = []
    for _stem, data in inference:
        if isinstance(data, list):
            gen_rows.extend(d for d in data if isinstance(d, dict))

    scores = []
    for _stem, data in evaluation:
        if isinstance(data, list):
            scores.extend(d for d in data if isinstance(d, dict))

    by_concept = {s.get("concept"): s for s in scores if s.get("concept")}

    rows = []
    for index, row in enumerate(gen_rows):
        concept = row.get("concept")
        scored = by_concept.get(concept)
        raw = (scored or {}).get("originality")
        try:
            value = float(str(raw).strip()[:1]) if raw not in (None, "") else None
        except ValueError:
            value = None
        rows.append({
            "sample_id": concept or index,
            "metric": "originality" if value is not None else None,
            "prompt": concept,
            "output": row.get("metaphors"),
            "eval_score": value,
        })
    return rows
