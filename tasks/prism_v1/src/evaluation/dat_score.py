"""DAT scoring: average pairwise semantic distance between generated nouns.

Lifted from ``neocoder_dat/src/utils/dat_score.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo) (itself from Olson et
al. 2021), with three changes:

* **gensim is imported lazily.** It is only needed for the word2vec path, but the
  import sat at module scope, so a missing gensim stopped the GloVe path -- the
  only path anyone uses -- from importing at all. Same shape of bug as the gated
  tokenizer in creativity_index.
* ``Model_old`` is gone. It was a byte-for-byte older copy of ``Model`` with no
  caller.
* ``distance()`` had a second, unreachable ``return``. Removed.

**What this computes, and how it differs from the paper.** The score is the mean
cosine distance over all pairs of the first ``minimum`` (7) valid unique words,
times 100 -- the original DAT definition. The paper's appendix instead describes
"average pairwise cosine similarity ... as D_0" with "DAT = 1 - D_0", which is a
different quantity on a different scale, and says nothing about the cut to 7
words. The code is what produced the numbers; the discrepancy is Joey's to
resolve.
"""
import itertools
import re

import numpy
import scipy.spatial.distance


class Model:
    """Word vectors restricted to a dictionary, plus the DAT score."""

    def __init__(self, model="glove.840B.300d.txt", dictionary="words.txt",
                 model_type="glove", pattern="^[a-z][a-z-]*[a-z]$"):
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))

        vectors = {}
        if model_type == "glove":
            with open(model, "r", encoding="utf8") as f:
                for line in f:
                    tokens = line.split(" ")
                    word = tokens[0]
                    if word in words:
                        vectors[word] = numpy.asarray(tokens[1:], "float32")
        elif model_type == "word2vec":
            from gensim.models import KeyedVectors  # only this path needs gensim
            loaded = KeyedVectors.load_word2vec_format(model, binary=True)
            for word in words:
                try:
                    vectors[word] = loaded[word]
                except KeyError:
                    pass
        elif model_type == "fasttext":
            import io
            with io.open(model, "r", encoding="utf-8", newline="\n", errors="ignore") as fin:
                for line in fin:
                    tokens = line.rstrip().split(" ")
                    if tokens[0] in words:
                        vectors[tokens[0]] = numpy.array([float(i) for i in tokens[1:]])
        else:
            raise ValueError("unknown model_type {!r}".format(model_type))

        self.vectors = vectors

    def validate(self, word):
        """Clean a word and return the first candidate spelling in the vocabulary."""
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        if len(clean) <= 1:
            return None

        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))
        for cand in candidates:
            if cand in self.vectors:
                return cand
        return None

    def distance(self, word1, word2):
        """Cosine distance (0 to 2) between two in-vocabulary words."""
        return scipy.spatial.distance.cosine(self.vectors[word1], self.vectors[word2])

    def dat(self, words, minimum=7):
        """Mean pairwise distance over the first `minimum` valid unique words, x100.

        Returns None when fewer than `minimum` words survive validation, which is
        how a response that did not produce enough usable nouns is distinguished
        from one that scored badly.
        """
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)

        if len(uniques) < minimum:
            return None
        subset = uniques[:minimum]

        distances = [self.distance(w1, w2) for w1, w2 in itertools.combinations(subset, 2)]
        return (sum(distances) / len(distances)) * 100
