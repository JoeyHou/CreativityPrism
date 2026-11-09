# Submit a result

Want your model on the [leaderboard](leaderboard.md)? Run it through the eight CreativityPrism tasks with the code in our [GitHub repo](https://github.com/joeyhou/creativityprism) and send us the results.

## Requirements

- `vllm` 0.7.2 (or >= 0.7.0), only needed if you serve the model locally
- Python 3.9 to 3.12
- CUDA >= 12.1

## 1. Set up

```bash
git clone https://github.com/joeyhou/creativityprism
cd creativityprism
bash registry/environments/setup.sh --env modern
```

## 2. Register your model

See what the runner already knows:

```bash
python task_runner/run.py --list-tasks
python task_runner/run.py --list-models
```

If your model isn't listed, add an entry for it in `registry/models.yaml` and use that name for `--model` below.

...

## 3. Try a dry run

This previews a run without loading a model or calling an API:

```bash
python task_runner/run.py \
    --task aut \
    --model <your-model> \
    --judge-model <judge> \
    --label smoke \
    --limit 5 \
    --dry-run
```

Drop `--dry-run` for a real smoke test. `--limit` must be a positive integer; leave it out to run the full dataset.

## 4. Run all eight tasks

The tasks are `ttcw`, `aut`, `creative_short_story`, `ttct`, `neocoder`, `dat`, `creative_math` and `creativity_index`.

```bash
for t in ttcw aut creative_short_story ttct neocoder dat creative_math creativity_index; do
  python task_runner/run.py --task "$t" --model "<your-model>" \
      --judge-model "<judge>" --label "<run-label>"
done
```

Use the same judge model as the paper so scores are comparable: ...

!!! warning "`creativity_index`"
    This task looks up n-grams in an external index, and the full configuration is not feasible against the public endpoint. See [the coverage caveat](#coverage-caveat-creativity_index). Say in your submission how you handled it (subset, your own index, or skipped).

## 5. Check your results

Make sure your run actually shows up in the shared table:

```bash
python -c "import sys; sys.path.insert(0,'result_analysis'); import loader; \
           print(loader.load_records(run_id='<run-label>'))"
```

## 6. Send it to us

Open a pull request or issue on the repo with:

- your model name, size, and a link to weights or API docs
- the judge model used
- the aggregated scores (Overall, Quality, Novelty, Diversity, and per-domain), in `[0, 1]`, in the same format as [`leaderboard.json`](assets/data/leaderboard.json)
- the raw outputs from your run, or a link to them

We will check the numbers and add them. ...

## Submit a new task

Have a creativity task that should be in the benchmark? See [Adding a task](adding-a-task.md).

## Data

- Download **leaderboard JSON**: [assets/data/leaderboard.json](assets/data/leaderboard.json)

### Notes

- Scores are in `[0, 1]` range in JSON; pages display as percentages.
- Add per-task breakdowns as additional JSON files (e.g., `assets/data/creative_writing.json`) and link them here.

### Coverage caveat: `creativity_index`

`creativity_index` scores a generation by looking up every n-gram span in an
external index (infini-gram). One document at one n-gram size is several hundred
HTTP round trips, and the full published configuration -- 100 documents x 3
domains x 8 n-gram sizes -- is on the order of 1.7 million lookups. That is not
feasible against the public endpoint, which rate-limits by IP.

**This task has therefore not been validated end to end at full scale in this
repository.** Inference runs normally; the scoring stage is verified on small
subsets. If you need the full configuration, plan on hosting your own index.