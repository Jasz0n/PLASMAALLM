"""Pinned embedding backend for KDP Stage 5 concept clustering.

Provides deterministic phrase vectors for merging paraphrases with little
or no lexical overlap. ``HashingEmbedder`` needs no ML deps (CI-safe);
``SentenceTransformerEmbedder`` uses a pinned HuggingFace model when the
``sentence-transformers`` package is installed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

PINNED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_THRESHOLD = 0.55
ST_EMBED_THRESHOLD = 0.72
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("a an and the of for to in on with between".split())


def _normalized_words(phrase: str) -> tuple[str, ...]:
    words: list[str] = []
    for token in _WORD.findall(phrase.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        words.append(token)
    return tuple(words)


class ConceptEmbedder(Protocol):
    """Maps concept phrases to unit-normalised embedding vectors."""

    def embed_phrases(self, phrases: list[str]) -> dict[str, tuple[float, ...]]:
        """Return one L2-normalised vector per distinct phrase."""


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two unit vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class HashingEmbedder:
    """Deterministic character/word hashing — no external dependencies."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed_phrases(self, phrases: list[str]) -> dict[str, tuple[float, ...]]:
        return {phrase: self._vector(phrase) for phrase in phrases}

    def _vector(self, phrase: str) -> tuple[float, ...]:
        text = phrase.lower()
        vec = [0.0] * self._dim
        for index in range(max(0, len(text) - 2)):
            gram = text[index : index + 3]
            bucket = int(hashlib.sha256(gram.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        words = _normalized_words(phrase)
        for token in words:
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 4.0
        for left, right in zip(words, words[1:], strict=False):
            bigram = f"{left}_{right}"
            bucket = int(hashlib.sha256(bigram.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 2.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return tuple(value / norm for value in vec)


class SentenceTransformerEmbedder:
    """Lazy-loaded pinned sentence-transformers model."""

    def __init__(self, model_id: str = PINNED_MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None

    def embed_phrases(self, phrases: list[str]) -> dict[str, tuple[float, ...]]:
        model = self._load()
        raw = model.encode(
            phrases,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return {phrase: tuple(float(x) for x in vector) for phrase, vector in zip(phrases, raw, strict=True)}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id)
        return self._model


def resolve_embedder(kind: str = "auto") -> ConceptEmbedder:
    """Pick an embedder: hashing, sentence-transformers, or auto-detect."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder()
    try:
        import sentence_transformers  # noqa: F401

        return SentenceTransformerEmbedder()
    except ImportError:
        return HashingEmbedder()


def embedding_threshold(embedder: ConceptEmbedder) -> float:
    """Default cosine threshold tuned for the embedder backend."""
    if isinstance(embedder, SentenceTransformerEmbedder):
        return ST_EMBED_THRESHOLD
    return DEFAULT_EMBED_THRESHOLD


def resolve_embedding_config() -> tuple[ConceptEmbedder | None, float]:
    """Read ``ALLM_KDP_EMBEDDINGS`` / ``ALLM_KDP_EMBED_BACKEND`` from the environment."""
    if os.environ.get("ALLM_KDP_EMBEDDINGS", "0") != "1":
        return None, DEFAULT_EMBED_THRESHOLD
    backend = os.environ.get("ALLM_KDP_EMBED_BACKEND", "auto")
    embedder = resolve_embedder(backend)
    threshold = float(os.environ.get("ALLM_KDP_EMBED_THRESHOLD", embedding_threshold(embedder)))
    return embedder, threshold
