"""Phased KEL learning order: books then workshops (M37)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

PhaseSource = Literal["book", "workshop"]

PHASE_ORDERS = {
    "books_then_workshops": ("book", "workshop"),
    "workshops_then_books": ("workshop", "book"),
    "books_only": ("book",),
}


@dataclass(frozen=True)
class LearningPhase:
    """One books-or-workshop segment of a KEL run."""

    source: PhaseSource
    iterations: int


def parse_learning_phases() -> tuple[LearningPhase, ...] | None:
    """Parse ``ALLM_KEL_PHASE_ORDER`` into phase segments."""
    order_key = os.environ.get("ALLM_KEL_PHASE_ORDER", "").strip().lower()
    if not order_key:
        return None
    if order_key not in PHASE_ORDERS:
        raise ValueError(
            f"unknown ALLM_KEL_PHASE_ORDER={order_key!r}; "
            f"expected one of {sorted(PHASE_ORDERS)}"
        )
    sources = PHASE_ORDERS[order_key]
    if order_key == "books_only":
        book_iters = int(
            os.environ.get("ALLM_KEL_BOOK_ITERS", os.environ.get("ALLM_ITERATIONS", "8"))
        )
        return (LearningPhase(source="book", iterations=book_iters),)
    book_iters = int(os.environ.get("ALLM_KEL_BOOK_ITERS", os.environ.get("ALLM_ITERATIONS", "2")))
    workshop_iters = int(
        os.environ.get("ALLM_KEL_WORKSHOP_ITERS", os.environ.get("ALLM_ITERATIONS", "2"))
    )
    counts = {"book": book_iters, "workshop": workshop_iters}
    return tuple(LearningPhase(source=source, iterations=counts[source]) for source in sources)


def discovery_order_from_phase() -> str | None:
    """Map phase order to Researcher discovery ordering."""
    order_key = os.environ.get("ALLM_KEL_PHASE_ORDER", "").strip().lower()
    if order_key == "books_then_workshops":
        return "books_first"
    if order_key == "books_only":
        return "books_first"
    if order_key == "workshops_then_books":
        return "workshops_first"
    explicit = os.environ.get("ALLM_DISCOVERY_ORDER", "").strip().lower()
    return explicit or None
