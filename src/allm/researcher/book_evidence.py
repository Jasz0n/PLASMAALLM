"""Build multimodal evidence rows from extracted book figures (M27)."""

from __future__ import annotations

import re

from allm.researcher.book_images import BookImageArtifact
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue

_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "they",
        "their",
        "which",
        "when",
        "where",
        "what",
        "will",
        "into",
        "about",
        "book",
        "page",
        "chapter",
        "figure",
    }
)
_TOKEN = re.compile(r"[a-z]{4,}")


def _page_excerpt(page_text: str, page_number: int) -> str:
    cleaned = " ".join(page_text.split())
    if cleaned:
        return cleaned[:240]
    return f"Book figure on page {page_number}"


def concept_hints_from_page_text(page_text: str, *, limit: int = 3) -> tuple[str, ...]:
    """Derive curriculum concept hints from surrounding page prose."""
    tokens = _TOKEN.findall(page_text.lower())
    ranked: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        ranked.append(token.replace("_", " "))
        if len(ranked) >= limit:
            break
    if ranked:
        return tuple(ranked)
    return ("plasma", "book diagram")


def build_book_synced_evidence(
    artifacts: tuple[BookImageArtifact, ...] | list[BookImageArtifact],
) -> tuple[SyncedEvidence, ...]:
    """Convert extracted book figures into Teacher-reviewable synced evidence."""
    rows: list[SyncedEvidence] = []
    for artifact in artifacts:
        excerpt = _page_excerpt(artifact.page_text, artifact.page_number)
        hints = concept_hints_from_page_text(artifact.page_text)
        rows.append(
            SyncedEvidence(
                source_id=artifact.book_id,
                timestamp_sec=float(artifact.page_number),
                transcript_excerpt=excerpt,
                visual=VisualCue(
                    description=f"Book diagram — {artifact.pdf_name} page {artifact.page_number}",
                    frame_path=artifact.image_path,
                    is_diagram=True,
                    tags=("book", "diagram", "keshe"),
                ),
                concept_hints=hints,
                confidence=0.82,
            )
        )
    return tuple(rows)
