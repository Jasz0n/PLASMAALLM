"""Stage 5 — Concept Normalization.

Merges semantically similar concept phrases deterministically:

1. explicit alias table (caller-supplied domain knowledge),
2. canonical keys (lowercase, stopwords stripped, crude de-plural),
3. token-overlap clustering (union-find, Jaccard >= 0.5).

"self attention mechanism" and "Self-Attention" land in one cluster;
its display name is the most frequent surface form (ties broken by
shortest, then alphabetical — fixed order, per KDP.md 7.5 determinism).

Embedding-based clustering is the planned upgrade for paraphrases with
no lexical overlap ("attention between tokens"); it must be pinned
(fixed model + weights) to keep the determinism guarantee.
"""

from __future__ import annotations

import re
from collections import Counter

from allm.kdp.embeddings import ConceptEmbedder, cosine_similarity
from allm.kdp.types import RawKnowledgeUnit

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("a an and the of for to in on with mechanism method".split())

CLUSTER_THRESHOLD = 0.5


def canonical_tokens(phrase: str) -> tuple[str, ...]:
    """Order-preserving, stopword-free, de-pluralised tokens."""
    tokens = []
    for token in _WORD.findall(phrase.lower()):
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tuple(tokens)


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller root wins
            ra, rb = sorted((ra, rb))
            self._parent[rb] = ra


def normalize(
    units: list[RawKnowledgeUnit],
    aliases: dict[str, str] | None = None,
    *,
    embedder: ConceptEmbedder | None = None,
    embed_threshold: float = 0.55,
) -> list[RawKnowledgeUnit]:
    """Return units with ``concept`` replaced by its cluster's display name."""
    alias_map = {k.lower(): v for k, v in (aliases or {}).items()}
    surface = [alias_map.get(u.concept.lower(), u.concept) for u in units]

    phrases = sorted(set(surface))
    keys = {p: canonical_tokens(p) for p in phrases}
    uf = _UnionFind(phrases)
    for i, a in enumerate(phrases):
        for b in phrases[i + 1 :]:
            if keys[a] == keys[b] or _jaccard(keys[a], keys[b]) >= CLUSTER_THRESHOLD:
                uf.union(a, b)

    if embedder is not None and phrases:
        vectors = embedder.embed_phrases(phrases)
        for i, a in enumerate(phrases):
            for b in phrases[i + 1 :]:
                if cosine_similarity(vectors[a], vectors[b]) >= embed_threshold:
                    uf.union(a, b)

    counts = Counter(surface)
    display: dict[str, str] = {}
    clusters: dict[str, list[str]] = {}
    for phrase in phrases:
        clusters.setdefault(uf.find(phrase), []).append(phrase)
    for members in clusters.values():
        best = max(members, key=lambda p: (counts[p], -len(p), [-ord(c) for c in p]))
        name = best.strip().title()
        for member in members:
            display[member] = name

    return [
        unit.model_copy(update={"concept": display[phrase]})
        for unit, phrase in zip(units, surface)
    ]
