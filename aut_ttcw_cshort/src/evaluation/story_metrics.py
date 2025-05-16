import spacy
from collections import Counter
from statistics import mean
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score, euclidean_sim, manhattan_sim
from sklearn.cluster import AgglomerativeClustering
from spacy_syllables import SpacySyllables
import benepar

# from utils import cache
def is_immutable(obj):
    return isinstance(obj, (str, int, float, bool, tuple, type(None)))
def cache(cache_dict):
    def decorator_cache(func):
        def wrapper(*args, **kwargs):
            if all(is_immutable(arg) for arg in args) and all(is_immutable(val) for val in kwargs.values()):
                key = (args, frozenset(kwargs.items()))
                if key in cache_dict:
                    return cache_dict[key]
                result = func(*args, **kwargs)
                cache_dict[key] = result
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_cache

DEF_EMB_MODEL = "thenlper/gte-large"
DEF_EMB_TYPE = "sentence_embedding"
DEF_DIST_FN = "cosine"
DEF_SPACY_LANG = "en_core_web_sm"
DEF_PREPROCESSING_ARGS = {
    "lower": True,
    "remove_punct": True,
    "remove_stopwords": True,
    "lemmatize": True,
    "dominant_k": None,
    "unique": True
}

SPACY_ENGINE_CACHE = {}
EMB_MODEL_CACHE = {}
EMBEDDING_CACHE = {}
SPACY_CACHE = {}

@cache(cache_dict=SPACY_ENGINE_CACHE)
def load_spacy_engine(language=DEF_SPACY_LANG, include_syllables=False, include_constituency=False):
    print(f"Loading spacy engine: {language}")
    engine = spacy.load(language)
    if include_syllables:
        engine.add_pipe("syllables", after="tagger")
    if include_constituency:
        engine.add_pipe('benepar', config={'model': 'benepar_en3'})
    return engine

@cache(cache_dict=EMB_MODEL_CACHE)
def load_emb_model(model=DEF_EMB_MODEL):
    print(f"Loading embedding model: {model}")
    return SentenceTransformer(model)

@cache(cache_dict=EMBEDDING_CACHE)
def get_embedding(text, model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE):
    emb_model = load_emb_model(model)
    
    output_value = "sentence_embedding"

    if "token" in emb_type:
        output_value = "token_embeddings"

    embeddings = emb_model.encode(text, output_value=output_value)
    
    if embeddings.ndim == 2:
        embeddings = embeddings.mean(axis=0)

    return embeddings

@cache(cache_dict=SPACY_CACHE)
def get_spacy_doc(text, include_syllables=False, include_constituency=False):
    spacy_engine = load_spacy_engine(include_syllables=include_syllables, include_constituency=include_constituency)
    return spacy_engine(text)

def compute_sem_dis(emb1, emb2, distance_fn=DEF_DIST_FN):
    if distance_fn == "cosine":
        return (1 - cos_sim(emb1, emb2)).item()
    elif distance_fn == "dot":
        return (1 - dot_score(emb1, emb2)).item()
    elif distance_fn == "euclidean":
        return (-euclidean_sim(emb1, emb2)).item()
    elif distance_fn == "manhattan":
        return (-manhattan_sim(emb1, emb2)).item()
    else:
        raise ValueError(f"Invalid distance function: {distance_fn}")

def get_sentences(text):
    doc = get_spacy_doc(text)
    return [sent.text for sent in doc.sents]

def get_words(text, lower=True, remove_punct=True, remove_stopwords=True, lemmatize=True, unique=True, dominant_k=None):
    doc = get_spacy_doc(text)
    tokens = [token for token in doc]

    if remove_punct:
        tokens = [token for token in tokens if not token.is_punct]
    
    if remove_stopwords:
        tokens = [token for token in tokens if not token.is_stop]

    words = [token.text for token in tokens]

    if lemmatize:
        words = [token.lemma_ for token in tokens]
    
    if lower:
        words = [word.lower() for word in words]
    
    if dominant_k is None or dominant_k == 0 or dominant_k >= len(words):
        if unique:
            return list(set(words))
        return words

    word_freq = Counter(words)

    return [w[0] for w in word_freq.most_common(dominant_k)]

def get_syllables(text):
    doc = get_spacy_doc(text, include_syllables=True)
    return [(token._.syllables, token._.syllables_count) for token in doc if token._.syllables_count]

def get_pos_tags(text, remove_punct=True):
    doc = get_spacy_doc(text)
    return [token.pos_ for token in doc if not (remove_punct and token.is_punct)]

def compute_text_embedding(text, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct",
                            preprocessing_args=DEF_PREPROCESSING_ARGS):
    if emb_strategy == "direct":
        return get_embedding(text, emb_model, emb_type)
    elif emb_strategy == "by_word":
        words = get_words(text, **preprocessing_args)
        return get_embedding(words, emb_model, emb_type)
    elif emb_strategy == "by_sentence":
        sentences = get_sentences(text)
        return get_embedding(sentences, emb_model, emb_type)
    else:
        raise ValueError(f"Invalid embedding strategy: {emb_strategy}")

def compute_avg_pairwise_distances(embeddings, distance_fn=DEF_DIST_FN):
    if len(embeddings) <= 1:
        return [0]

    avg_pairwise_distances = []
    for i in range(len(embeddings)):
        pairwise_distances = []
        for j in range(len(embeddings)):
            if i != j:
                distance = compute_sem_dis(embeddings[i], embeddings[j], distance_fn)
                pairwise_distances.append(distance)
        avg_pairwise_distances.append(mean(pairwise_distances))
    return avg_pairwise_distances

def compute_avg_sem_dis(text, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    words = get_words(text, **preprocessing_args)
    embeddings = [get_embedding(word, emb_model, emb_type) for word in words]
    return mean(compute_avg_pairwise_distances(embeddings, distance_fn))

def compute_inverse_homogenization(texts, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct", 
                                   distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    text_embeddings = [compute_text_embedding(text, emb_model, emb_type, emb_strategy, preprocessing_args) for text in texts]
    return compute_avg_pairwise_distances(text_embeddings, distance_fn)

def compute_novelty(texts, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, 
                    distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    corpus = "".join(texts)
    corpus_avg_sem_dis = compute_avg_sem_dis(corpus, emb_model, emb_type, distance_fn, preprocessing_args)

    novelty_scores = []

    for text in texts:
        text_avg_sem_dis = compute_avg_sem_dis(text, emb_model, emb_type, distance_fn, preprocessing_args)
        novelty_scores.append(2 * abs(text_avg_sem_dis - corpus_avg_sem_dis))
    
    return novelty_scores

def compute_theme_uniqueness(texts, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, emb_strategy="direct",
                             cluster_linkage="ward", cluster_distance_threshold=0.5, preprocessing_args=DEF_PREPROCESSING_ARGS):
    text_embeddings = [compute_text_embedding(text, emb_model, emb_type, emb_strategy, preprocessing_args) for text in texts]
    clustering = AgglomerativeClustering(n_clusters=None, linkage=cluster_linkage, distance_threshold=cluster_distance_threshold)
    cluster_labels = clustering.fit_predict(text_embeddings)
    cluster_freq = Counter(cluster_labels)
    return [1 / cluster_freq[cluster_labels[i]] for i in range(len(texts))], cluster_labels

def compute_dsi(text, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    return compute_avg_sem_dis(text, emb_model, emb_type, distance_fn, preprocessing_args)

def compute_surprise(text, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    sentences = get_sentences(text)

    if len(sentences) <= 1:
        return 0, []

    sentence_avg_sem_distances = [compute_avg_sem_dis(sentence, emb_model, emb_type, distance_fn, preprocessing_args) for sentence in sentences]
    raw_surprises = [abs(sentence_avg_sem_distances[i] - sentence_avg_sem_distances[i - 1]) for i in range(1, len(sentence_avg_sem_distances))]

    return (2 / len(raw_surprises)) * sum(raw_surprises), raw_surprises

# def compute_n_gram_diversity(text, max_n_gram=5, remove_punct=True):
#     words = get_words(text, lower=True, remove_punct=remove_punct, remove_stopwords=False, lemmatize=False, unique=False)
#     all_n_grams = []

#     for n in range(1, max_n_gram + 1):
#         all_n_grams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    
#     all_n_gram_freqs = [Counter(n_grams) for n_grams in all_n_grams]
#     n_gram_diversity = [len(n_gram_freqs) / len(n_grams) for n_grams, n_gram_freqs in zip(all_n_grams, all_n_gram_freqs)]

#     return n_gram_diversity, all_n_gram_freqs

# def compute_pos_diversity(text, max_n_gram=5, remove_punct=True):
#     pos_tags = get_pos_tags(text, remove_punct=remove_punct)
#     all_pos_tags = []

#     for n in range(1, max_n_gram + 1):
#         all_pos_tags.append([tuple(pos_tags[i:i + n]) for i in range(len(pos_tags) - n + 1)])
    
#     all_pos_tag_freqs = [Counter(p_tags) for p_tags in all_pos_tags]
#     pos_tag_diversity = [len(pos_freqs) / len(p_tags) for p_tags, pos_freqs in zip(all_pos_tags, all_pos_tag_freqs)]

#     return pos_tag_diversity, all_pos_tag_freqs

def _get_dep_paths(token):
    if not list(token.children):
        return [(token.dep_,)]
    paths = []
    for child in token.children:
        child_paths = _get_dep_paths(child)
        for path in child_paths:
            paths.append((token.dep_,) + path)
    return paths
    
def compute_dependency_complexity(text):
    # see labels https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    # see https://euroslajournal.org/articles/10.22599/jesla.63
    clause_labels = ["conj", "cc", "preconj", "ccomp", "xcomp", "acl", "relcl", "advcl"]
    dep_num_clauses = []
    dep_paths = []
    sentences = get_sentences(text)

    for sentence in sentences:
        doc = get_spacy_doc(sentence)
        num_clauses = 0
        sent_path_counter = Counter()
        for token in doc:
            paths = _get_dep_paths(token)
            sent_path_counter.update(paths)
            if token.dep_ in clause_labels:
                num_clauses += 1
        dep_num_clauses.append(num_clauses)
        dep_paths.append(sent_path_counter)
    
    return dep_paths, dep_num_clauses

def compute_pos_complexity(text):
    sentences = get_sentences(text)
    pos_complexity = {
        "NOUN": [],
        "VERB": [],
        "ADJ": [],
        "ADV": [],
        "PRON": [],
        "DET": [],
        "ADP": []
    }

    for sentence in sentences:
        pos_tags = get_pos_tags(sentence)
        pos_freq = Counter(pos_tags)
        num_words = len(pos_tags)
        for pos in pos_complexity:
            pos_complexity[pos].append(pos_freq.get(pos, 0) / num_words)
    
    return pos_complexity

def compute_flesch_readability_scores(text):
    # see https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    num_words = len(get_words(text, lower=True, remove_punct=False, remove_stopwords=False, lemmatize=False, unique=False))
    num_sentences = len(get_sentences(text))
    num_syllables = sum([n_syllables for _, n_syllables in get_syllables(text)])
    flesch_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    flesch_kincaid = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    return flesch_ease, flesch_kincaid

def compute_constituency_complexity(text):
    text = re.sub(r"\s+", " ", text)
    def _get_height(sent):
        if not list(sent._.children):
            return 1
        return 1 + max([_get_height(child) for child in sent._.children])
    doc = get_spacy_doc(text, include_constituency=True)
    return [_get_height(sent) for sent in doc.sents]

def compute_interestingness(target_text, source_text, emb_model=DEF_EMB_MODEL, emb_type=DEF_EMB_TYPE, distance_fn=DEF_DIST_FN, preprocessing_args=DEF_PREPROCESSING_ARGS):
    source_avg_sem_dis = compute_avg_sem_dis(source_text, emb_model=emb_model, emb_type=emb_type, distance_fn=distance_fn, preprocessing_args=preprocessing_args)
    target_avg_sem_dis = compute_avg_sem_dis(target_text, emb_model=emb_model, emb_type=emb_type, distance_fn=distance_fn, preprocessing_args=preprocessing_args)
    return 2 * abs(source_avg_sem_dis - target_avg_sem_dis)













def compute_n_gram_diversity(text, max_n_gram=5, remove_punct=True):
    words = get_words(text, lower=True, remove_punct=remove_punct, remove_stopwords=False, lemmatize=False, unique=False)
    all_n_grams = []

    for n in range(1, max_n_gram + 1):
        all_n_grams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    
    all_n_gram_freqs = [Counter(n_grams) for n_grams in all_n_grams]
    n_gram_diversity = [len(n_gram_freqs) / len(n_grams) for n_grams, n_gram_freqs in zip(all_n_grams, all_n_gram_freqs)]

    return n_gram_diversity, all_n_gram_freqs

def compute_pos_diversity(text, max_n_gram=5, remove_punct=True):
    pos_tags = get_pos_tags(text, remove_punct=remove_punct)
    all_pos_tags = []

    for n in range(1, max_n_gram + 1):
        all_pos_tags.append([tuple(pos_tags[i:i + n]) for i in range(len(pos_tags) - n + 1)])
    
    all_pos_tag_freqs = [Counter(p_tags) for p_tags in all_pos_tags]
    pos_tag_diversity = [len(pos_freqs) / len(p_tags) for p_tags, pos_freqs in zip(all_pos_tags, all_pos_tag_freqs)]

    return pos_tag_diversity, all_pos_tag_freqs
