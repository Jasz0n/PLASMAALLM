"""Cross-source concept alignment between workshop and book packages (M29)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.types import KnowledgePackage

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({"the", "and", "from", "that", "this", "with", "have", "been"})


class ConceptAlignment(BaseModel):
    """One workshop ↔ book concept correspondence."""

    model_config = ConfigDict(frozen=True)

    workshop_concept: str
    book_concept: str
    overlap_score: float = Field(ge=0.0, le=1.0)
    status: Literal["aligned"] = "aligned"


class CrossSourceReport(BaseModel):
    """Alignment summary across workshop transcripts and Keshe books."""

    model_config = ConfigDict(frozen=True)

    workshop_package_id: str
    book_package_id: str
    alignments: tuple[ConceptAlignment, ...] = ()
    workshop_only: tuple[str, ...] = ()
    book_only: tuple[str, ...] = ()
    aligned_count: int = 0
    summary: str = ""


def _concept_tokens(name: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(name.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _concept_tokens(left)
    right_tokens = _concept_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    shared = left_tokens & right_tokens
    if not shared:
        return 0.0
    return round(len(shared) / min(len(left_tokens), len(right_tokens)), 4)


def align_workshop_and_book(
    workshop: KnowledgePackage,
    book: KnowledgePackage,
    *,
    min_overlap: float = 0.35,
) -> CrossSourceReport:
    """Find concept token overlaps between workshop and book packages."""
    workshop_names = [concept.name for concept in workshop.concepts]
    book_names = [concept.name for concept in book.concepts]
    alignments: list[ConceptAlignment] = []
    matched_workshop: set[str] = set()
    matched_book: set[str] = set()

    for workshop_name in workshop_names:
        best_name = ""
        best_score = 0.0
        for book_name in book_names:
            score = _overlap_score(workshop_name, book_name)
            if score > best_score:
                best_score = score
                best_name = book_name
        if best_name and best_score >= min_overlap:
            alignments.append(
                ConceptAlignment(
                    workshop_concept=workshop_name,
                    book_concept=best_name,
                    overlap_score=best_score,
                )
            )
            matched_workshop.add(workshop_name)
            matched_book.add(best_name)

    workshop_only = tuple(name for name in workshop_names if name not in matched_workshop)
    book_only = tuple(name for name in book_names if name not in matched_book)
    aligned_count = len(alignments)
    summary = (
        f"{aligned_count} aligned concept(s) between {workshop.provider} and {book.provider}"
    )
    if workshop_only:
        summary += f"; {len(workshop_only)} workshop-only"
    if book_only:
        summary += f"; {len(book_only)} book-only"

    return CrossSourceReport(
        workshop_package_id=workshop.id,
        book_package_id=book.id,
        alignments=tuple(sorted(alignments, key=lambda row: -row.overlap_score)),
        workshop_only=workshop_only[:12],
        book_only=book_only[:12],
        aligned_count=aligned_count,
        summary=summary,
    )


def find_packages_by_provider(
    packages: tuple[KnowledgePackage, ...] | list[KnowledgePackage],
) -> tuple[KnowledgePackage | None, KnowledgePackage | None]:
    """Return (workshop_package, book_package) when both exist."""
    workshop = book = None
    for package in packages:
        if package.provider == "kids-workshops" and workshop is None:
            workshop = package
        if package.provider == "keshe-books" and book is None:
            book = package
    return workshop, book
