"""Book vs workshop training samples and phased visual delivery (M37)."""

from __future__ import annotations

from typing import Literal

from allm.data.base import Sample
from allm.researcher.types import KnowledgePackage
from allm.students.model_student import ModelStudent
from allm.teacher.student_visual_delivery import (
    deliver_visual_notes,
    packages_for_provider,
)

LearningSource = Literal["book", "workshop", "all"]

BOOK_PROVIDER = "keshe-books"
WORKSHOP_PROVIDER = "kids-workshops"


def samples_from_book_packages(
    packages: tuple[KnowledgePackage, ...] | list[KnowledgePackage],
    *,
    topic: str,
    limit: int | None = None,
) -> list[Sample]:
    """Turn Keshe book package definitions into study samples."""
    rows: list[Sample] = []
    for package in packages:
        if package.provider != BOOK_PROVIDER:
            continue
        package_topic = package.curriculum_topic or topic
        for index, (term, definition) in enumerate(package.definitions):
            if not term.strip() or not str(definition).strip():
                continue
            rows.append(
                Sample(
                    id=f"book_{package.id}_{index}",
                    input=f"What is {term}?",
                    target=str(definition),
                    metadata={
                        "topic": package_topic,
                        "source": "book",
                        "provider": package.provider,
                        "pin": True,
                    },
                )
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def filter_workshop_samples(samples: list[Sample]) -> list[Sample]:
    """Keep pooled workshop curriculum samples (exclude book-tagged rows)."""
    return [row for row in samples if row.metadata.get("source") != "book"]


def filter_workshop_delta_samples(
    samples: list[Sample],
    packages: tuple[KnowledgePackage, ...] | list[KnowledgePackage],
    *,
    limit: int | None = None,
) -> list[Sample]:
    """Prefer workshop-only concepts not already aligned to the book graph."""
    from allm.researcher.cross_source import align_workshop_and_book, find_packages_by_provider

    workshop_pkg, book_pkg = find_packages_by_provider(packages)
    if workshop_pkg is None or book_pkg is None:
        return samples[:limit] if limit is not None else samples
    report = align_workshop_and_book(workshop_pkg, book_pkg)
    delta_terms = [term.lower() for term in report.workshop_only]
    if not delta_terms:
        return samples[:limit] if limit is not None else samples
    filtered: list[Sample] = []
    for row in samples:
        haystack = f"{row.input} {row.target}".lower()
        if any(term in haystack for term in delta_terms):
            filtered.append(row)
    rows = filtered if filtered else samples
    if limit is not None:
        return rows[:limit]
    return rows


def deliver_visuals_for_source(
    researcher: object,
    student: ModelStudent,
    source: LearningSource,
) -> int:
    """Deliver Teacher-approved visuals for one source phase."""
    if not hasattr(researcher, "stored_packages"):
        return deliver_visuals_from_researcher_fallback(researcher, student, source)
    packages = tuple(researcher.stored_packages())
    if source == "book":
        exports = packages_for_provider(packages, BOOK_PROVIDER, topic=student.specialty)
    elif source == "workshop":
        exports = packages_for_provider(packages, WORKSHOP_PROVIDER, topic=student.specialty)
    else:
        from allm.teacher.student_visual_delivery import deliver_visuals_from_researcher

        return deliver_visuals_from_researcher(researcher, student)
    return deliver_visual_notes(student, exports)


def deliver_visuals_from_researcher_fallback(
    researcher: object,
    student: ModelStudent,
    source: LearningSource,
) -> int:
    """Fallback when only ``student_visual_packages`` is available."""
    if not hasattr(researcher, "student_visual_packages"):
        return 0
    exports = researcher.student_visual_packages(topic=student.specialty)
    if source == "all":
        return deliver_visual_notes(student, exports)
    # Without package linkage, deliver all in "all" mode only.
    return deliver_visual_notes(student, exports)
