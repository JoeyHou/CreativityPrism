# Adding a task

Derived from actually doing it: a deliberately awkward task was written into
`tasks/contrib_metaphor/` and integrated without touching a single central file.
Every rule below earned its place by something breaking first.

## What you write — four files, no central edits

| File | What it is |
|---|---|
| `tasks/<your_task>/` | your code, in whatever layout you already have |
| `registry/tasks/<your_task>.yaml` | the declaration the runner reads |
| `registry/adapters/<your_task>.sh` | translates the runner's flags into your CLI |
| `tasks/<your_task>/parser.py` | turns your outputs into rows for the shared table |

**You do not edit the runner, the loader, or any gate.** The task list is read
from `registry/tasks/*.yaml`; anything that needs to know about tasks asks there.
It used to be written out by hand in six places, and two of those copies were
silently stale.

Don't want to fork? Point `CREATIVITYPRISM_INCLUDE_PATH` at a directory of your
own YAML files and the runner picks them up.

## The YAML

```yaml
name: your_task                 # must match the filename and --task
display_name: "Your Task"
folder: tasks/your_task
adapter: registry/adapters/your_task.sh
environment: modern             # or your own; add registry/environments/<name>.yml
uses_judge: true                # false = --judge-model is a documented no-op
stages: [inference, eval]       # omit for both; [inference] for generation-only
limit_supported: true           # false = the runner rejects --limit
loader_parser: tasks/your_task/parser.py:parse
```

`loader_parser` takes a file path *or* a dotted module. The path form exists so
your parser can live beside your code: the alternative was adding it to our
loader, which is exactly the central edit this is meant to avoid.

## The adapter

Read `registry/adapters/contrib_metaphor.sh` — it is kept as a worked example of
a task that follows none of our conventions.

**Use `run_stage <your command>`.** It runs the command, or prints it under
`CREATIVITYPRISM_ADAPTER_DRYRUN=1` so the gate can check your adapter without a
GPU, a key, or a paid call.

**Announce what you produced**, with a path, not a directory if you can help it:

```bash
emit_artifact inference "$TASK_DIR/results/whatever_you_called_it.csv"
```

If you announce a *directory*, the runner records the data files under it and
subtracts any file another kind claimed. That rule exists because announcing an
output directory as `inference` used to hand the analysis loader the evaluation
files sitting beside it, and the rows came back doubled and mislabelled.

**If your task names its own output at run time** (a timestamp, say), take a
marker before the stage and use `newest_since`:

```bash
MARKER="$(mktemp)"; touch "$MARKER"
run_stage python scripts/gen.py --num-examples "$LIMIT"
CSV="$(newest_since "$MARKER" results/*/gen_*.csv)"
```

Globbing for "the most recent" is wrong the moment two runs overlap.

## Models: use `model_hub`, and pass the canonical name

If your task hardcodes a provider, this is the one change you will have to make
to your own source. Nothing outside the task can fix it: the alternative is
copying a credential into a second file, which is both a bad idea and against
this repo's rules.

```python
from model_hub import get_client
client = get_client(args.model_name, temperature=0.9, max_tokens=200)
text = client.generate(prompt)
```

In the adapter, pass **`$MODEL` and `$JUDGE` directly**. Do *not* call
`lookup_alias`: that maps a canonical name to the per-task alias the built-in
bundle's own model table wants, and feeding it to `model_hub` produces
`Model 'gpt-4.1-mini' not in registry/models.yaml` while `GPT4.1-mini` sits right
there in the registry.

## The parser

```python
def parse(inference, evaluation):
    """Each argument is [(file_stem, data), ...]. Return row dicts."""
    return [{"sample_id": ..., "metric": ..., "prompt": ...,
             "output": ..., "eval_score": ...}]
```

`eval_score` of `None` means "this row ran but was not scored" — keep those rows.
Dropping them makes the generation count look like the evaluation count and
hides how much the evaluator skipped.

## Before you open the PR

```bash
bash task_runner/tests/test_adapters.sh   # your adapter, dry-run, no cost
python task_runner/run.py --task your_task --model GPT4.1-mini \
    --judge-model GPT4.1-mini --label smoke --limit 2
python -c "import sys; sys.path.insert(0,'result_analysis'); import loader; \
           print(loader.load_records(run_id='smoke'))"
```

The last one is the real check. An adapter can exit 0, announce artifacts, and
still contribute nothing to the shared table — that has happened here more than
once, and only reading the rows catches it.
