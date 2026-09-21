# CreativityPrism

A cross-domain benchmark for LLM creativity: **eight tasks** spanning divergent
thinking, creative writing, mathematical creativity, code creativity, and
originality measurement.

Paper: *CreativityPrism: A Cross-Domain Evaluation Framework for Large Language
Model Creativity* (see [CITATION.cff](CITATION.cff)). Published results are on
the [leaderboard](docs/leaderboard.md); they were produced by the paper's runs,
not re-run from this repository.

## Install

```bash
git clone https://github.com/JoeyHou/CreativityPrism && cd CreativityPrism
pip install -e ".[tasks]"                    # add ",open" for local vLLM inference
cp api_keys.example.json api_keys.json       # then fill in your keys
```

`api_keys.json` is gitignored. Point `CREATIVITYPRISM_API_KEYS` elsewhere if you
keep credentials outside the repo. On a cluster, `bash registry/environments/setup.sh
--env modern` builds the conda environment the tasks were developed against.

Installing puts a `creativityprism` command on your path; it is the same thing as
`python task_runner/run.py` and works from any directory.

## Data

The data is not in this repository. It is published as
[joeyhou/creativityprism_v1](https://huggingface.co/datasets/joeyhou/creativityprism_v1),
versioned separately from the code so that a score is determined by a stated
(dataset revision, harness version, judge config) triple. Pull it once:

```bash
creativityprism data pull
```

That downloads the dataset into `tasks/prism_v1/data/processed/` and then fills in
the one task whose text we are not allowed to redistribute: Creativity Index comes
from a repository with no licence, so the dataset ships its item ids and a pointer
to each item upstream, and the fetch step downloads the text from the original
source, at a pinned commit, onto your machine. Use of that text is governed by the
original authors' terms.

```bash
creativityprism data status
```

shows what is on disk and at which dataset revision. `CREATIVITYPRISM_DATA_DIR`
puts the data somewhere else -- another disk, or one copy shared between several
checkouts.

To look at the data without the harness:

```python
from datasets import load_dataset
load_dataset("joeyhou/creativityprism_v1", "aut")
```

`python release/build.py` rebuilds the Hub payload from the data root, which is how
a new dataset revision is cut.

## Run

One task, one model, through the runner:

```bash
creativityprism --task aut --model GPT4.1-mini --judge-model GPT4.1-mini \
                --label myrun --limit 10
```

- `--task all` runs every task; `--list-tasks` lists them.
- `--list-models` shows the registered model names.
- `--limit N` caps the sample count. `--dry-run` prints the commands only.
- `--inference-only` / `--eval-only` split the two stages.
- `--slurm` submits one job per task instead of running locally.

**Check first, then run.** A generation for a retired model, or a judge whose
verdict cannot be parsed, costs hours before it shows up in a report:

```bash
python -m model_hub.smoke --models GPT4.1-mini --judges GPT4.1-mini
```

Non-zero exit means don't start. Each of the failure modes it distinguishes
(unreachable, retired, bad key, no credentials, unparseable verdict) needs a
different fix.

## Read the results

Everything a run produced is linked under `outputs/<label>/<task>/<model>/`
(`CREATIVITYPRISM_OUTPUT_ROOT` moves that), and
one call turns all of it into one table:

```python
import sys; sys.path.insert(0, "result_analysis")
import loader
df = loader.load_outputs(run_id="myrun")        # pandas
rows = loader.load_records(run_id="myrun")      # plain dicts, no pandas needed
```

One row per scored unit, never a mean: `sample_id`, `metric`, `eval_score`,
`prompt`, `output`. An `eval_score` of `None` means the row ran but was not
scored — keep those, they are how you see what the evaluator skipped.

## The eight tasks

| Task | What it measures |
|---|---|
| `aut` | alternative uses for everyday objects |
| `ttct` | Torrance Tests of Creative Thinking |
| `dat` | divergent association, scored by GloVe distance |
| `ttcw` | Torrance Test of Creative Writing, binary expert rubrics |
| `creative_short` | short-story generation, n-gram and embedding diversity |
| `creative_math` | novel solutions to solved problems |
| `neocoder` | code creativity under denial prompting |
| `creativity_index` | overlap with a pretraining corpus, via an n-gram index |

Two caveats, both real:

- **`dat`** needs `glove.840B.300d.txt` (5.3 GB, not shipped). Set
  `CREATIVITYPRISM_GLOVE_PATH`.
- **`creativity_index`** scores against the public infini-gram index, which
  rate-limits by IP. The published configuration is ~1.7M lookups and is not
  feasible there; **this task has not been validated at full scale here.** See
  [docs/data.md](docs/data.md).

## Layout

```
task_runner/     the CLI (run.py); tests/ holds the gates, fixtures/ their configs
registry/        what the runner reads: one YAML per task, adapters, model aliases
tasks/prism_v1/  the eight tasks: one class per task under src/{inference,evaluation}
tasks/           other task bundles live beside it; contrib_metaphor is a worked example
model_hub/       one way to call a model, whichever provider it is; plus smoke.py
result_analysis/ loader.py — every task's outputs as one table
docs/            the published site, and adding-a-task.md
research/        data cleaning and human annotation; not needed to run the benchmark
```

The runner never reads a task's files. An adapter announces what it produced,
the runner links it into `outputs/` and records it in `metadata.json`, and the
loader reads only that. A task can keep any layout it likes.

## Add a task

Four files, none of them ours: your code, a registry YAML, an adapter, and a
parser. See **[docs/adding-a-task.md](docs/adding-a-task.md)** —
`tasks/contrib_metaphor/` is a deliberately awkward task kept as a worked example.

## Tests

```bash
for g in task_runner/tests/test_*.sh; do bash "$g"; done
```

No GPU, no API key, no paid call.

## License

Apache 2.0. The earlier README is kept at
[docs/archive/README-v1.md](docs/archive/README-v1.md).
