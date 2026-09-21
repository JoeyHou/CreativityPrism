# Data

- Download **leaderboard JSON**: [assets/data/leaderboard.json](assets/data/leaderboard.json)

## Notes
- Scores are in `[0, 1]` range in JSON; pages display as percentages.
- Add per-task breakdowns as additional JSON files (e.g., `assets/data/creative_writing.json`) and link them here.

## Coverage caveat: `creativity_index`

`creativity_index` scores a generation by looking up every n-gram span in an
external index (infini-gram). One document at one n-gram size is several hundred
HTTP round trips, and the full published configuration -- 100 documents x 3
domains x 8 n-gram sizes -- is on the order of 1.7 million lookups. That is not
feasible against the public endpoint, which rate-limits by IP.

**This task has therefore not been validated end to end at full scale in this
repository.** Inference runs normally; the scoring stage is verified on small
subsets. If you need the full configuration, plan on hosting your own index.
