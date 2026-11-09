# Method

CreativityPrism evaluates LLM creativity with **8 tasks** from **3 domains**, scored by **17 automatic metrics** that are grouped into **3 dimensions**: quality, novelty, and diversity. Every metric is automatic, and each LLM-judged metric was checked against human annotations.

- **Domains:** Creative Writing, Divergent Thinking, Logical Reasoning
- **Datasets:** [CreativityPrism dataset](https://huggingface.co/datasets/joeyhou/creativityprism_v1)
- **Metrics:** 17, grouped into **Quality**, **Novelty** and **Diversity**
- **Code:** [GitHub](https://github.com/joeyhou/creativityprism); to evaluate your own model, see [Submit a result](submit.md)

## Tasks

| Task | Domain | The model is asked to | Size |
|---|---|---|---|
| AUT (Alternative Uses Test) | Divergent thinking | list creative alternative uses for an everyday object | 21 objects × 5 rounds |
| DAT (Divergent Association Task) | Divergent thinking | name 10 nouns that are as unrelated to each other as possible | 100 rounds |
| TTCT (Torrance Tests of Creative Thinking) | Divergent thinking | answer open-ended questions, e.g. what if humans lost the ability to sleep | 500 questions (5 types × 100) |
| TTCW (Torrance Test of Creative Writing) | Creative writing | write a New Yorker-style story from a plot summary | 12 plots |
| Creative Short Story | Creative writing | write a story of at most five sentences that uses three given words in unconventional ways | 10 word triples |
| Creativity Index | Creative writing | continue a prefix from a book, poem or speech | 300 (100 per genre) |
| NeoCoder | Logical reasoning | solve a coding problem under constraints, e.g. no HashMap | 198 problems |
| Creative Math | Logical reasoning | solve a competition math problem, where novelty is judged against reference solutions | 373 problems |

## Three dimensions

The taxonomy follows the verbal TTCT, which scores fluency, flexibility and originality. We map these to quality, diversity and novelty, and keep novelty and diversity separate rather than merging them into one "originality" score.

Think of the model's output distribution, the space of valid answers, and the space of ordinary answers:

- **Quality**: how much of the output falls inside the valid answer space, i.e. whether it does what the task asks. Examples: whether NeoCoder code runs and solves the problem, and whether a Creative Math solution is correct.
- **Novelty**: how far outputs reach beyond ordinary answers, i.e. how rare a single output is. Examples: Creativity Index L-Uniqueness (n-gram overlap with web text), and NeoCoder Divergence@0 (techniques absent from human solutions).
- **Diversity**: how widely the model explores different valid answers across generations. Examples: the DAT score (semantic distance between the nouns) and TTCT flexibility.

Tasks from different domains can sit in the same dimension. Each domain defines these ideas differently, and having several metrics per dimension covers more of what creativity means. The metric-to-dimension mapping is shown in the overview figure on the [home page](index.md#overview).

## Inference

- **Models:** 17 LLMs. Frontier-scale models (GPT, Claude, Gemini, DeepSeek) are run through their APIs. Open models under 80B (Mistral/Mixtral, Qwen, OLMo, Llama) run locally with vLLM 0.7.2.
- **Sampling:** temperature 0.75 and max 4096 tokens for all tasks, except Creativity Index (temperature 1.0, max 288 tokens) and Creative Math (temperature 0.0, max 2000 tokens). Outputs longer than the max are truncated.
- **Post-processing:** output is trimmed so judges score the creative content only. Preambles such as "Here's the story" are removed. AUT keeps the first ten lines, Creative Short Story asks for `[START]`/`[END]` markers, and TTCT outputs are stripped of formatting and politeness text to avoid length bias in the judge.
- **Runs:** the main results come from a single run per model. A 5-run check on AUT and TTCW shows stable scores (see the paper's appendix).

## Scoring

All metrics are rescaled to [0, 1], then averaged in two steps so that tasks with many metrics don't dominate a dimension:

```text
normalized   = (raw - min) / (max - min)            # per metric, using the task's min and max
task_score   = mean(normalized metrics of that task in the dimension)
dimension    = mean(task_score over the tasks in that dimension)   # quality, novelty, diversity
overall      = (quality + novelty + diversity) / 3
```

Domain scores (creative writing, divergent thinking, logical reasoning) are computed the same way. The overall score is only a convenience for comparing models. The right weighting of the three dimensions depends on the application, so we recommend picking the dimension and domain closest to your use case. The ranking is stable under three other aggregation schemes, with Spearman ρ ≥ .92.

## LLM judges

Five tasks (AUT, TTCW, TTCT, NeoCoder, Creative Math) use an LLM judge for at least one metric. Each judge was validated in two steps:

1. **Human agreement.** Annotators (10 CS PhD students and faculty, three per item) rated sampled model outputs. We kept only metrics with Fleiss' κ ≥ 0.4. TTCW uses the original paper's expert annotations.
2. **Judge vs. humans.** We ran the Alternative Annotator Test (Calderon et al., 2025) and required the judge to align with the human consensus better than the held-out human in more than half of the cases. For the objective Creative Math correctness metric, we check accuracy against humans instead.

The judge is chosen in this order: Qwen2.5-72B, then GPT-4.1, then the average of the two, then the setup from the original task paper. The first one that passes is used.

| Task | Judge |
|---|---|
| AUT | GPT-4.1 |
| NeoCoder | GPT-4.1 |
| TTCW | Qwen2.5-72B |
| TTCT | GPT-4.1 and Qwen2.5-72B |
| Creative Math (novelty) | Claude 3.7 Sonnet, GPT-4.1 and Gemini 2.0 Flash |
| Creative Math (correctness) | Claude 3.7 Sonnet (91% agreement with humans) |

Metrics that did not meet the bar were removed. TTCT drops the story-writing and just-suppose question types and the originality metric, and TTCW keeps only the metrics that passed.

## Known limitations

- Text only. Domains without reliable automatic evaluation, such as scientific discovery, are not covered.
- The human annotation sets are small and the annotators are CS researchers rather than creativity experts. Judge choice may add bias.
- Models released later may have an advantage on metrics that reward text that differs from existing content, such as Creativity Index and Creative Math, because of data contamination.
- Creativity Index at the full published scale needs a self-hosted n-gram index. See [Data](data.md#coverage-caveat-creativity_index).

For the prompts, per-task metrics and annotation details, see the [paper](https://arxiv.org/abs/2510.20091) (Appendices C, D and F).