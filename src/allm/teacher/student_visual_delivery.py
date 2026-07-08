"""Deliver Teacher-approved visual packages into student study memory (M25)."""

from __future__ import annotations

from allm.memory.episodic import EpisodicMemory
from allm.researcher.multimodal_types import StudentVisualPackage
from allm.researcher.types import KnowledgePackage
from allm.students.model_student import ModelStudent


def packages_for_topic(
    packages: tuple[KnowledgePackage, ...] | list[KnowledgePackage],
    topic: str,
) -> tuple[StudentVisualPackage, ...]:
    """Collect student-safe visual exports matching a curriculum topic."""
    topic_key = topic.strip().lower()
    exports: list[StudentVisualPackage] = []
    seen: set[str] = set()
    for package in packages:
        for export in package.student_visual_packages:
            export_topic = (export.curriculum_topic or package.curriculum_topic or "").lower()
            if export_topic and export_topic != topic_key:
                continue
            if export.export_id in seen:
                continue
            seen.add(export.export_id)
            exports.append(export)
    return tuple(exports)


def packages_for_provider(
    packages: tuple[KnowledgePackage, ...] | list[KnowledgePackage],
    provider: str,
    *,
    topic: str | None = None,
) -> tuple[StudentVisualPackage, ...]:
    """Collect student exports from one knowledge provider."""
    scoped = [package for package in packages if package.provider == provider]
    if topic is None:
        exports: list[StudentVisualPackage] = []
        seen: set[str] = set()
        for package in scoped:
            for export in package.student_visual_packages:
                if export.export_id in seen:
                    continue
                seen.add(export.export_id)
                exports.append(export)
        return tuple(exports)
    merged: list[StudentVisualPackage] = []
    for export in packages_for_topic(scoped, topic):
        merged.append(export)
    return tuple(dict.fromkeys(merged))


def visual_study_notes(export: StudentVisualPackage) -> tuple[tuple[str, str], ...]:
    """Convert one student visual export into studyable Q/A pairs."""
    concept = export.concept_name or "visual concept"
    notes: list[tuple[str, str]] = []

    if export.concept_description:
        notes.append(
            (
                f"What is {concept}?",
                export.concept_description,
            )
        )

    for index, image in enumerate(export.images, start=1):
        notes.append(
            (
                f"Describe visual {index} for {concept}.",
                image,
            )
        )

    if export.diagram:
        notes.append(
            (
                f"What does the diagram show for {concept}?",
                export.diagram,
            )
        )

    for index, explanation in enumerate(export.explanations, start=1):
        notes.append(
            (
                f"Explain {concept} (part {index}).",
                explanation,
            )
        )

    if export.experiment:
        notes.append(
            (
                f"What experiment relates to {concept}?",
                export.experiment,
            )
        )

    for question in export.questions:
        notes.append((question, f"Review the approved visual materials for {concept}."))

    return tuple(notes)


def deliver_visual_notes(
    student: ModelStudent,
    exports: tuple[StudentVisualPackage, ...],
) -> int:
    """Inject approved visual study notes into one student's memory."""
    delivered = 0
    for export in exports:
        for prompt, answer in visual_study_notes(export):
            student.study(prompt, answer)
            delivered += 1
    return delivered


def deliver_visuals_from_researcher(researcher: object, student: ModelStudent) -> int:
    """Deliver stored student visual packages for the student's specialty."""
    if not hasattr(researcher, "student_visual_packages"):
        return 0
    exports = researcher.student_visual_packages(topic=student.specialty)
    return deliver_visual_notes(student, exports)


def count_visual_notes_delivered(memory: EpisodicMemory, student_id: str) -> int:
    """Sum visual study notes delivered across loop iterations."""
    total = 0
    for episode in memory.recall(actor=student_id, kind="observation"):
        if "visual study note" not in episode.summary.lower():
            continue
        total += int(episode.detail.get("visual_notes", 0))
    return total
