---
pretty_name: CreativityPrism v1
license: other
license_name: per-task-upstream-licences
license_link: LICENSE
language:
  - en
task_categories:
  - text-generation
tags:
  - creativity
  - benchmark
  - llm-evaluation
  - divergent-thinking
  - creative-writing
size_categories:
  - 1K<n<10K
configs:
  - config_name: aut
    data_files: data/aut/aut_push_skipped.json
  - config_name: ttct
    data_files: data/ttct/ttct.json
  - config_name: creative_math
    data_files: data/creative_math/creative_math.json
  - config_name: creative_short
    data_files: data/creative_short_story/all.json
  - config_name: neocoder
    data_files: data/neocoder/NeoCoder.json
  - config_name: creativity_index
    data_files: data/creativity_index/*_prompt.json
  - config_name: ttcw_questions
    data_files: data/ttcw/ttcw_questions.json
---

# CreativityPrism v1 — benchmark data

Inputs for **CreativityPrism**, a cross-domain benchmark for LLM creativity: the
eight tasks of `tasks/prism_v1` in the harness, which lives at
[github.com/JoeyHou/CreativityPrism](https://github.com/JoeyHou/CreativityPrism).

This repository holds **data only** — no scoring code, no judge prompt templates,
no results. Those are versioned separately so that a score is determined by a
stated triple: dataset revision, harness version, judge configuration.

## Getting it

```python
from datasets import load_dataset
aut = load_dataset("joeyhou/creativityprism_v1", "aut")
```

Most tasks need more than their prompt file — NeoCoder executes against test
cases, DAT scores against a word list, TTCW judges against a rubric. To run the
benchmark, take the whole repository rather than a single config:

```python
from huggingface_hub import snapshot_download
snapshot_download("joeyhou/creativityprism_v1", repo_type="dataset",
                  local_dir="creativityprism_v1")
```

Then fill in the Creativity Index text (see below):

```bash
python creativityprism_v1/scripts/fetch_upstream.py
```

`data/` has the same layout as `tasks/prism_v1/data/processed/` in the harness.

## Why Creativity Index needs a fetch step

The Creativity Index prompts come from
[GXimingLu/creativity_index](https://github.com/GXimingLu/creativity_index),
which does not currently carry a license, so we cannot redistribute them. The
`creativity_index` config still has all 582 rows, with their ids and order, but
`input.text` is `null`. Each row's `meta_data.source_ref` names the upstream file,
row, and pinned commit it comes from, plus that file's sha256.

`scripts/fetch_upstream.py` downloads those files from the original source at
that commit, checks the checksums, and writes the text into your local copy.
It needs only the Python standard library, and re-running it does nothing once
the text is there. Use of that text is governed by the original authors' terms.
If a checksum does not match, the script stops: upstream has changed since we
pinned it, and scores computed from different text would not be comparable.

## Item format

Most tasks share three keys:

| Key | What it holds |
|---|---|
| `meta_data` | `dataset`, `id`, and task-specific fields |
| `input` | the prompt; `text` for most tasks, `text_basic` / `text_cot` / `text_instructive` where a task has prompt variants. `null` until fetched for `creativity_index` |
| `output` | a reference where one exists, `null` for open-ended generation |

`input` and `output` are **task-specific below the top level** — TTCT carries
three prompt variants, Creative Math carries reference solutions, Creativity
Index carries only a continuation prefix. That is why each task is its own
config rather than one table: there is no single schema that fits all eight
without flattening away what distinguishes them.

## What is here

| Config / file | Items | Role |
|---|---|---|
| `aut` (`aut_push_skipped.json`) | 63 | Alternative Uses items. `aut_push_all.json` (75) is the unfiltered pool; `aut_demos.json` holds judge few-shot demos |
| `ttct` | 700 | Torrance Tests of Creative Thinking, 100 items in each of 7 question types |
| `creative_math` | 373 | problems with reference solutions, for novel-solution generation |
| `creative_short` | 10 | short-story prompts |
| `neocoder` | 199 | competitive-programming problems with per-round denial statements. `test_cases_annotated.json` (499) is executed against; `human_solution_techniques.json` is the divergence reference |
| `creativity_index` | 582 | continuation prompts: book 100, poem 247, speech 235 (id prefix gives the domain). **Text not included; run `scripts/fetch_upstream.py`** |
| `ttcw_questions` | 14 | the binary expert rubric TTCW judges against. `original.json` holds the 12 story plots (the New Yorker stories themselves are linked, not included); `all_stories.json` holds 36 GPT-4 / GPT-3.5 / Claude stories from the TTCW paper, 8 of which the judge sees as few-shot demos listed in `ttcw_few_shot.json` |
| `dat_words.txt` | 175,276 | the noun vocabulary DAT scores against |

**DAT has no prompt file**: it is one fixed instruction, sampled repeatedly, so
the instruction lives in the harness and only its vocabulary is here.

`research/` holds artifacts that are *not* used to run the benchmark: the TTCW
paper's expert annotations (672 verdicts, 2016 with explanations) and its
GPT-3.5 / GPT-4 / Claude generations. They are kept because judge–human
agreement is checkable against them.

## Provenance and licensing

Every task reuses material published with someone else's paper. That material
stays under its own licence, which is reproduced in `licenses/`; `provenance.yaml`
carries the same information in machine-readable form, including what we changed
for each task.

| Task | Paper | Data from | Licence |
|---|---|---|---|
| `aut` | [Pushing the Limits of GPT's Creativity for Alternative Uses](https://kar.kent.ac.uk/101551/) | [massivetexts/llm_aut_study](https://github.com/massivetexts/llm_aut_study) (judge demos) | MIT — `licenses/aut.txt` |
| `ttct` | [Assessing and Understanding Creativity in Large Language Models](https://arxiv.org/abs/2401.12491) | shared with us by the authors | published with their permission |
| `ttcw` | [Art or Artifice?](https://arxiv.org/abs/2309.14556) | [salesforce/creativity_eval](https://github.com/salesforce/creativity_eval) | BSD-3-Clause — `licenses/ttcw.txt` |
| `dat` | [Divergent Association Task](https://openreview.net/forum?id=BpibUh0aB3) | [jayolson/divergent-association-task](https://github.com/jayolson/divergent-association-task) | MIT — `licenses/dat.txt` |
| `creative_math` | [Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems](https://arxiv.org/abs/2410.18336) | [JunyiYe/CreativeMath](https://github.com/JunyiYe/CreativeMath) | MIT — `licenses/creative_math.txt` |
| `creative_short` | [Evaluating Creative Short Story Generation in Humans and Large Language Models](https://arxiv.org/abs/2411.02316) | [mismayil/creative-story-gen](https://github.com/mismayil/creative-story-gen) | MIT — `licenses/creative_short.txt` |
| `neocoder` | [Benchmarking Language Model Creativity](https://arxiv.org/abs/2407.09007) | [JHU-CLSP/NeoCoder](https://github.com/JHU-CLSP/NeoCoder) | Apache-2.0 — `licenses/neocoder.txt` |
| `creativity_index` | [AI as Humanity's Salieri](https://arxiv.org/abs/2410.04265) | [GXimingLu/creativity_index](https://github.com/GXimingLu/creativity_index) | no licence — **text not included, fetched** |

Each task's row in `provenance.yaml` states how we changed the upstream material:
every task was reformatted to the row schema described above, and several use a
subset of what upstream published.

**`ttct`**: the prompts were shared with us directly by the authors rather than
taken from a public release, and are published here with their permission. Please
cite their paper if you use them.

**`creativity_index`**: see
[Why Creativity Index needs a fetch step](#why-creativity-index-needs-a-fetch-step).

## Our own licence

The row schema, item ids, task metadata and judge configuration are ours and are
released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The
harness code is Apache-2.0. Upstream material is governed by the licences in the
table above, not by ours.

## Citation

See [CITATION.cff](https://github.com/JoeyHou/CreativityPrism/blob/main/CITATION.cff).
