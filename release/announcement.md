# Launch copy

Ready-to-post announcement text for X and LinkedIn. Numbers come from the current
paper (arXiv 2510.20091 v3): 8 tasks, 3 domains, 3 dimensions, 17 models, the ~15%
frontier-vs-open gap, and novelty's weak-to-negative correlation with the other
metrics.

**Before posting**

- The Hub dataset is private until someone flips it. A link to it 404s for
  everyone else until then.
- No venue is claimed anywhere below. Add one if the paper has been accepted.
- Fill in the co-author handles in the last post of the thread.

---

## X / Twitter

**1/**

Is your LLM actually creative, or just fluent?

We built CreativityPrism: 8 tasks across 3 domains, every output scored on quality, novelty, and diversity. Then we ran 17 frontier and open models through it.

The headline: creativity doesn't transfer. 🧵

**2/**

Frontier models (DeepSeek-V3, GPT-4.1, Claude 3 Sonnet) lead open models by ~15% on creative writing and logical reasoning.

On divergent thinking — the classic psychology tasks — that lead disappears. Open models hold their own.

**3/**

Being good at one kind of creativity says little about the others.

Novelty is the clearest case: it correlates weakly, sometimes negatively, with quality and diversity. A single "creativity score" hides more than it shows.

**4/**

The judges are automatic and validated against human annotations, so the whole thing runs without a human in the loop.

Eight tasks, one command:

creativityprism --task all --model <your model>

**5/**

Everything is open:

📄 Paper: arxiv.org/abs/2510.20091
💻 Code: github.com/JoeyHou/CreativityPrism
🤗 Data: huggingface.co/datasets/joeyhou/creativityprism_v1
📊 Leaderboard: joeyhou.github.io/CreativityPrism/

**6/**

With @[co-author handles] — and thanks to the authors of the eight original tasks, whose work this builds on directly.

---

## LinkedIn

Large language models write fluently. That is not the same as being creative — and the difference is now measurable.

Our new benchmark, CreativityPrism, evaluates LLM creativity across 8 tasks and 3 domains (divergent thinking, creative writing, logical reasoning), scoring every generation on three dimensions: quality, novelty, and diversity. We evaluated 17 frontier and open models.

Three things stood out:

→ Frontier models lead open models by roughly 15% on creative writing and logical reasoning — but hold no meaningful advantage on divergent thinking, the domain post-training has largely ignored.

→ Creativity does not generalize. Strength in one dimension or domain rarely predicts the others, and novelty correlates weakly or even negatively with the rest.

→ Evaluation is fully automatic, with LLM judges validated against human annotations, so the benchmark scales past what human evaluation can cover.

Everything is public: the harness, the data (with per-task provenance and upstream licences), and a live leaderboard.

Paper: https://arxiv.org/abs/2510.20091
Code: https://github.com/JoeyHou/CreativityPrism
Leaderboard: https://joeyhou.github.io/CreativityPrism/

Grateful to my co-authors and to the authors of the eight tasks we built on.

#LLM #AIResearch #Benchmarking #Creativity #MachineLearning

---

## Platform notes

- Every post in the thread is under 280 characters. X previews only the first
  link it finds, so the order in post 5 decides which card appears.
- LinkedIn truncates at roughly 210 characters, which is why the first line
  stands alone as the hook. LinkedIn also demotes posts with outbound links;
  moving the three links into the first comment and leaving "links in the
  comments" in the body is the safer version.
- Suggested image: the three-dimension figure or the overall-performance bars
  from `docs/assets/img/`. Alt text, e.g. "Bar chart of 17 language models scored
  on quality, novelty and diversity."
