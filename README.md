# CreativityPrism
## Codebase Structure 

- `/data_cleaning`: all data access, cleaning and formating code
    - `/processed`: cleaned data for each task
    - `/raw_data`: raw data before cleaning
- `/tasks`: all CreativityPrism tasks
    - `/aut_ttcw_csshort`: evaluation codebase for [ttcw](https://arxiv.org/abs/2309.14556), [aut](https://kar.kent.ac.uk/101551/1/Pushing_the_Limits_of_GPT_s_Creativity_for_Alternative_Uses_and_Torrence_Tests.pdf), and [creative_short_story](https://arxiv.org/pdf/2411.02316)
    - `/neocoder_dat`: evaluation codebase for [neocoder](https://arxiv.org/pdf/2407.09007), and [dat](https://openreview.net/forum?id=BpibUh0aB3)
    - `/ttct`: evaluation codebase for [ttct](https://arxiv.org/abs/2401.12491)
    - `/math_n_index`: evaluation codebase for [creative_math](https://arxiv.org/pdf/2410.18336) and [creativity_index](https://arxiv.org/abs/2410.04265)
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
bash scripts/setup_envs.sh --env modern
```

Inspect available tasks and models:

```bash
python runner/run.py --list-tasks
python runner/run.py --list-models
```

Preview a run without loading models or calling an API:

```bash
python runner/run.py \
    --task aut \
    --model GPT4.1 \
    --judge-model GPT4.1-mini \
    --label smoke \
    --limit 5 \
    --dry-run
```

`--limit` must be a positive integer. Omit it to run the full task dataset.

See `claude_docs/WORKFLOW.md` for the current workflow and
`claude_docs/RESTRUCTURING_PLAN.md` for the phased architecture plan.