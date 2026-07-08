"""Distill enriched multimodal evidence into Teacher-ready visual briefs."""

from __future__ import annotations

import re
from collections import defaultdict

from allm.kdp.types import content_hash
from allm.researcher.multimodal_types import DistilledVisualBrief, SyncedEvidence
from allm.researcher.types import KnowledgePackage, PackageConcept

_EXPERIMENT_VECTORS = frozenset({"rotation", "oscillation", "translation"})
_TOKEN = re.compile(r"[a-z0-9]+")


def _normalize_concept(text: str) -> str:
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return "workshop visual"
    return " ".join(tokens[:4])


def _group_key(row: SyncedEvidence) -> str:
    if row.source_id.startswith("book:"):
        return f"{row.source_id}:page:{int(row.timestamp_sec)}"
    if row.object_identity_id:
        return row.object_identity_id
    if row.motion_track_id:
        return row.motion_track_id
    if row.concept_hints:
        return f"concept:{row.concept_hints[0].lower()}"
    return f"{row.source_id}:{row.timestamp_sec:.0f}"


def _image_description(row: SyncedEvidence) -> str | None:
    visual = row.visual
    if visual is None:
        return None
    if visual.caption:
        return visual.caption.strip()
    if visual.description:
        return visual.description.strip()
    return None


def _diagram_summary(rows: list[SyncedEvidence]) -> str | None:
    parts: list[str] = []
    for row in rows:
        visual = row.visual
        if visual is None or not visual.is_diagram:
            continue
        if visual.diagram_labels:
            parts.append(f"Labels: {', '.join(visual.diagram_labels[:6])}")
        if visual.ocr_text:
            parts.append(visual.ocr_text.strip()[:160])
    if not parts:
        for row in rows:
            visual = row.visual
            if visual is None:
                continue
            if visual.diagram_labels or visual.ocr_text:
                if visual.diagram_labels:
                    parts.append(f"Labels: {', '.join(visual.diagram_labels[:6])}")
                if visual.ocr_text:
                    parts.append(visual.ocr_text.strip()[:160])
    if not parts:
        return None
    return " | ".join(dict.fromkeys(parts))


def _build_explanations(rows: list[SyncedEvidence]) -> tuple[str, ...]:
    explanations: list[str] = []
    for row in sorted(rows, key=lambda item: -item.confidence):
        excerpt = row.transcript_excerpt.strip()
        if excerpt and excerpt not in explanations:
            explanations.append(excerpt[:220])
        for note in (row.continuity_summary, row.identity_summary):
            if note and note not in explanations:
                explanations.append(note[:220])
        visual = row.visual
        if visual is not None and visual.analytics_summary and visual.analytics_summary not in explanations:
            explanations.append(visual.analytics_summary[:220])
    return tuple(explanations[:2])


def _build_questions(concept_name: str, rows: list[SyncedEvidence], *, limit: int) -> tuple[str, ...]:
    questions: list[str] = []
    for row in rows:
        for hint in row.concept_hints:
            question = f"What did the workshop show about {hint.strip()}?"
            if question not in questions:
                questions.append(question)
    if not questions:
        questions.append(f"What visual evidence supports understanding of {concept_name}?")
    motion_vectors = {
        row.visual.motion_vector
        for row in rows
        if row.visual is not None and row.visual.motion_vector
    }
    if "rotation" in motion_vectors or "oscillation" in motion_vectors:
        questions.append(f"How does motion in the demo relate to {concept_name}?")
    if any(row.visual and row.visual.is_diagram for row in rows):
        questions.append(f"Which diagram elements best explain {concept_name}?")
    return tuple(dict.fromkeys(questions))[:limit]


def _build_experiment(concept_name: str, rows: list[SyncedEvidence]) -> str | None:
    vectors = {
        row.visual.motion_vector
        for row in rows
        if row.visual is not None and row.visual.motion_vector
    }
    if not vectors & _EXPERIMENT_VECTORS:
        return None
    vector = sorted(vectors & _EXPERIMENT_VECTORS)[0]
    return (
        f"Observe a tabletop {concept_name} demonstration with {vector} motion. "
        "Note field interactions without fuel or conventional motors."
    )


def _concept_from_rows(rows: list[SyncedEvidence]) -> tuple[str, str]:
    for row in rows:
        if row.concept_hints:
            name = _normalize_concept(row.concept_hints[0])
            description = row.concept_hints[0].strip()
            return name, description
    visual = rows[0].visual
    if visual is not None:
        return _normalize_concept(visual.description), visual.description.strip()[:180]
    return "workshop visual", rows[0].transcript_excerpt.strip()[:180]


def distill_visual_group(
    rows: list[SyncedEvidence],
    *,
    max_images: int = 3,
    max_questions: int = 5,
) -> DistilledVisualBrief | None:
    """Distill one evidence group into a Teacher handoff brief."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: -row.confidence)
    concept_name, concept_description = _concept_from_rows(ordered)

    images: list[str] = []
    for row in ordered:
        description = _image_description(row)
        if description and description not in images:
            images.append(description[:240])
        if len(images) >= max_images:
            break

    source_refs = tuple(
        dict.fromkeys(
            f"{row.source_id}@{row.timestamp_sec:.0f}s" for row in ordered
        )
    )
    confidence = round(
        sum(row.confidence for row in ordered) / len(ordered),
        4,
    )
    diagram = _diagram_summary(ordered)
    explanations = _build_explanations(ordered)
    experiment = _build_experiment(concept_name, ordered)
    questions = _build_questions(concept_name, ordered, limit=max_questions)

    notes_parts = [
        "Distilled for Teacher review — students receive selected assets only.",
    ]
    if ordered[0].identity_summary:
        notes_parts.append(ordered[0].identity_summary[:160])
    elif ordered[0].continuity_summary:
        notes_parts.append(ordered[0].continuity_summary[:160])

    brief_id = "dvis_" + content_hash(concept_name, *source_refs[:3])
    source_kind = "book" if ordered[0].source_id.startswith("book:") else "workshop"
    if source_kind == "book":
        notes_parts.append("Book figure — approve before student delivery.")
    return DistilledVisualBrief(
        brief_id=brief_id,
        concept_name=concept_name,
        concept_description=concept_description,
        images=tuple(images),
        diagram_summary=diagram,
        explanations=explanations,
        experiment_prompt=experiment,
        questions=questions,
        source_refs=source_refs,
        source_kind=source_kind,
        evidence_confidence=confidence,
        teacher_notes=" ".join(notes_parts),
    )


def distill_visual_evidence(
    rows: list[SyncedEvidence],
    *,
    max_images: int = 3,
    max_questions: int = 5,
) -> tuple[DistilledVisualBrief, ...]:
    """Distill synced evidence rows into Teacher-ready visual briefs."""
    grouped: dict[str, list[SyncedEvidence]] = defaultdict(list)
    for row in rows:
        if row.visual is None:
            continue
        grouped[_group_key(row)].append(row)

    briefs: list[DistilledVisualBrief] = []
    for group_rows in grouped.values():
        brief = distill_visual_group(
            group_rows,
            max_images=max_images,
            max_questions=max_questions,
        )
        if brief is not None:
            briefs.append(brief)
    briefs.sort(key=lambda brief: -brief.evidence_confidence)
    return tuple(briefs)


def align_briefs_to_concepts(
    briefs: tuple[DistilledVisualBrief, ...],
    concepts: tuple[PackageConcept, ...],
) -> tuple[DistilledVisualBrief, ...]:
    """Attach package concept names to briefs when token overlap is strong."""
    if not concepts:
        return briefs
    aligned: list[DistilledVisualBrief] = []
    for brief in briefs:
        brief_tokens = set(_TOKEN.findall(brief.concept_name.lower()))
        best_name = brief.concept_name
        for concept in concepts:
            concept_tokens = set(_TOKEN.findall(concept.name.lower()))
            if brief_tokens & concept_tokens:
                best_name = concept.name
                break
        if best_name != brief.concept_name:
            aligned.append(
                brief.model_copy(
                    update={
                        "concept_name": best_name,
                        "concept_description": brief.concept_description or best_name,
                    }
                )
            )
        else:
            aligned.append(brief)
    return tuple(aligned)


def briefs_for_provider(
    briefs: tuple[DistilledVisualBrief, ...],
    provider: str,
) -> tuple[DistilledVisualBrief, ...]:
    """Return briefs whose source refs match one knowledge provider."""
    if provider == "keshe-books":
        return tuple(brief for brief in briefs if brief.source_kind == "book")
    if provider == "kids-workshops":
        return tuple(brief for brief in briefs if brief.source_kind != "book")
    return briefs


def attach_distilled_visuals(
    package: KnowledgePackage,
    briefs: tuple[DistilledVisualBrief, ...],
) -> KnowledgePackage:
    """Attach distilled visual briefs to one knowledge package."""
    if not briefs:
        return package
    aligned = align_briefs_to_concepts(briefs, package.concepts)
    merged = tuple(dict.fromkeys(package.distilled_visual_briefs + aligned))
    return package.model_copy(update={"distilled_visual_briefs": merged})
