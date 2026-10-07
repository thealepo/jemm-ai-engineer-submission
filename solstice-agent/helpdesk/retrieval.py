"""Knowledge-base retrieval: embedding recall + deterministic BM25 rerank."""

from collections import Counter
import math
import os
import re
from difflib import SequenceMatcher

from . import config, embeddings
from .chunking import chunk_text

# Kept for the legacy evidence audit. Production search no longer calls it.
RERANK_PROMPT = """You are a relevance judge for a support knowledge base.

Query: {query}

Passage: {passage}

RELEVANCE: rate how relevant the passage is to the query on a scale of 0-10.
Respond with a single integer."""

TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_documents():
    docs = []
    for name in sorted(os.listdir(config.KB_DIR)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(config.KB_DIR, name)) as f:
            docs.append(f.read())
    return docs


def build_index():
    index = []
    for doc in load_documents():
        for chunk in chunk_text(doc):
            index.append((chunk, embeddings.embed(chunk)))
    return index


def dedupe(chunks):
    """Drop near-duplicate chunks so the rerank stage isn't wasted on repeats."""
    kept = []
    for c in chunks:
        dup = False
        for k in kept:
            if SequenceMatcher(None, c, k).ratio() > 0.9:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _tokens(text):
    return TOKEN_RE.findall(text.lower())


def rank_bm25(query, candidates, corpus):
    """Return candidates in BM25 order, preserving dense order for score ties."""
    if not candidates or not corpus:
        return list(candidates)

    corpus_tokens = [_tokens(document) for document in corpus]
    document_frequency = Counter()
    for tokens in corpus_tokens:
        document_frequency.update(set(tokens))

    average_length = sum(len(tokens) for tokens in corpus_tokens) / float(len(corpus_tokens))
    average_length = average_length or 1.0
    query_terms = list(dict.fromkeys(_tokens(query)))
    document_count = len(corpus_tokens)

    def score(document):
        tokens = _tokens(document)
        frequencies = Counter(tokens)
        length_ratio = len(tokens) / average_length
        total = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            containing = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (document_count - containing + 0.5) / (containing + 0.5)
            )
            denominator = frequency + config.BM25_K1 * (
                1.0 - config.BM25_B + config.BM25_B * length_ratio
            )
            total += inverse_frequency * frequency * (config.BM25_K1 + 1.0) / denominator
        return total

    scored = [(score(candidate), candidate) for candidate in candidates]
    scored.sort(key=lambda row: row[0], reverse=True)
    return [candidate for _, candidate in scored]


def search(llm, query, k=None):
    k = k or config.TOP_K
    index = build_index()
    qv = embeddings.embed(query)

    scored = [(embeddings.similarity(qv, vec), chunk) for chunk, vec in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [chunk for _, chunk in scored[: config.RERANK_CANDIDATES]]
    candidates = dedupe(candidates)
    reranked = rank_bm25(query, candidates, [chunk for chunk, _ in index])
    return reranked[:k]
