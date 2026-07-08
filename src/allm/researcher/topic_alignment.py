"""Align Researcher package concepts with curriculum / pool topics."""

from __future__ import annotations

from allm.data.base import Sample
from allm.kdp.concept_quality import is_noisy_concept


def pool_topics_from_samples(samples: list[Sample]) -> set[str]:
    """Distinct ``metadata['topic']`` values from a sample pool."""
    topics: set[str] = set()
    for sample in samples:
        topic = sample.metadata.get("topic")
        if topic:
            topics.add(str(topic))
    return topics


def align_recommendation_topic(
    concept_name: str,
    *,
    curriculum_topic: str | None,
    catalog_topics: set[str] | None = None,
) -> str:
    """Map a package concept to the topic the planner and pool understand.

    Workshop KDP emits fragment labels (``The Beauty Of It``); labelled
  samples use coarse topics (``kids-plasma``). When a package declares
    ``curriculum_topic`` or the concept already exists in the catalog,
    recommendations use that curriculum key instead of the raw label.
    """
    catalog = catalog_topics or set()
    stripped = concept_name.strip()
    if stripped in catalog:
        return stripped
    if curriculum_topic and curriculum_topic in catalog:
        return curriculum_topic
    if curriculum_topic and is_noisy_concept(stripped):
        return curriculum_topic
    if curriculum_topic:
        return curriculum_topic
    return stripped
