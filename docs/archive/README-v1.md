# CreativityPrism
## Codebase Structure 

- `/research`: research-stage assets; not needed to run the benchmark
    - `/data_cleaning`: data access, cleaning and formatting code (`/processed`, `/raw_data`)
    - `/human_annotation`: MTurk corpora and the human-agreement analysis
- `/tasks/prism_v1`: all eight tasks, behind two entrypoints (`run_inference.py`, `run_evaluation.py`)
  and one class per task under `src/inference/` and `src/evaluation/`:
  [ttcw](https://arxiv.org/abs/2309.14556),
  [aut](https://kar.kent.ac.uk/101551/1/Pushing_the_Limits_of_GPT_s_Creativity_for_Alternative_Uses_and_Torrence_Tests.pdf),
  [creative_short_story](https://arxiv.org/pdf/2411.02316),
  [ttct](https://arxiv.org/abs/2401.12491),
  [neocoder](https://arxiv.org/pdf/2407.09007),
  [dat](https://openreview.net/forum?id=BpibUh0aB3),
  [creative_math](https://arxiv.org/pdf/2410.18336),
  [creativity_index](https://arxiv.org/abs/2410.04265).
  Adding a task is one line in each entrypoint's dispatch dict plus one class.
- `/tasks/archive`: the four original per-paper bundles the above was ported from.
  Untracked (see `.gitignore`) and kept only so a published number can be traced to
  the code that produced it. Nothing in the runner points here.
- `/registry`: task metadata, model aliases, adapters, and environment declarations
- `/runner`: unified benchmark CLI and behavior tests
- `/runs`: reproducible run configurations

> Note: the CS4 task was removed from the benchmark during the v2 restructuring.

## Requirement
- `vllm`: [0.7.2](https://docs.vllm.ai/en/v0.7.2/getting_started/installation/index.html) (or >= 0.7.0)
- `Python`: 3.9 – 3.12
- `cuda`: >= 12.1

## Get Started

Create the declared Conda environment:

```bash
bash registry/environments/setup.sh --env modern
```

Inspect available tasks and models:

```bash
python task_runner/run.py --list-tasks
python task_runner/run.py --list-models
```

Preview a run without loading models or calling an API:

```bash
python task_runner/run.py \
    --task aut \
    --model GPT4.1 \
    --judge-model GPT4.1-mini \
    --label smoke \
    --limit 5 \
    --dry-run
```

`--limit` must be a positive integer. Omit it to run the full task dataset.

See [docs/adding-a-task.md](docs/adding-a-task.md) for contributing a task, and
`python task_runner/run.py --help` for running one.