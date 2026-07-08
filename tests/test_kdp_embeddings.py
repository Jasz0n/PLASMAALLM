"""Tests for KDP embedding-based concept clustering."""

from allm.kdp.embeddings import HashingEmbedder, cosine_similarity
from allm.kdp.normalization import normalize
from allm.kdp.types import RawKnowledgeUnit, SpanRef


def _unit(concept: str) -> RawKnowledgeUnit:
    return RawKnowledgeUnit(
        type="concept",
        concept=concept,
        content=f"{concept} is explained here.",
        span=SpanRef(doc="d", start=0, end=10),
        context="x",
    )


class _MockEmbedder:
    """Forces paraphrase pairs into one cluster."""

    def embed_phrases(self, phrases: list[str]) -> dict[str, tuple[float, ...]]:
        paraphrase = {
            "self attention mechanism",
            "attention between tokens",
        }
        vectors: dict[str, tuple[float, ...]] = {}
        for phrase in phrases:
            if phrase in paraphrase:
                vectors[phrase] = (1.0, 0.0)
            else:
                vectors[phrase] = (0.0, 1.0)
        return vectors


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()
    first = embedder.embed_phrases(["Plasma", "Magnetic Field"])
    second = embedder.embed_phrases(["Plasma", "Magnetic Field"])
    assert first == second


def test_near_duplicate_phrases_share_embedding_signal() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed_phrases(["magnetic field", "the magnetic fields"])
    similarity = cosine_similarity(vectors["magnetic field"], vectors["the magnetic fields"])
    assert similarity >= 0.55


def test_normalize_merges_near_duplicates_with_hashing() -> None:
    units = [_unit("magnetic field"), _unit("the magnetic fields")]
    merged = normalize(units, embedder=HashingEmbedder(), embed_threshold=0.55)
    assert merged[0].concept == merged[1].concept


def test_normalize_merges_paraphrases_with_mock_embedder() -> None:
    units = [_unit("self attention mechanism"), _unit("attention between tokens")]
    merged = normalize(units, embedder=_MockEmbedder(), embed_threshold=0.5)
    assert merged[0].concept == merged[1].concept


def test_normalize_without_embedder_keeps_distinct_paraphrases() -> None:
    units = [_unit("self attention mechanism"), _unit("attention between tokens")]
    plain = normalize(units)
    assert plain[0].concept != plain[1].concept
