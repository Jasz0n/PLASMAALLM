"""Route incoming concepts to specialist students by domain fit."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from allm.students.identity import StudentIdentity, domain_fit


class ConceptAssignment(BaseModel):
    """One concept routed to one student."""

    model_config = ConfigDict(frozen=True)

    concept: str
    student_id: str
    fit: float = Field(ge=0.0, le=1.0)
    reason: str


class IngestRouter:
    """Assign KDP concepts to students whose mission matches."""

    def __init__(self, identities: Iterable[StudentIdentity], *, seed: int = 0) -> None:
        self._identities = tuple(identities)
        self._seed = seed

    def route(self, concepts: Iterable[str]) -> list[ConceptAssignment]:
        """Return every (concept, student) pair with non-zero mission fit."""
        assignments: list[ConceptAssignment] = []
        for concept in concepts:
            for identity in self._identities:
                fit, reason = domain_fit(concept, identity, seed=self._seed)
                if fit > 0.0:
                    assignments.append(
                        ConceptAssignment(
                            concept=concept,
                            student_id=identity.student_id,
                            fit=fit,
                            reason=reason,
                        )
                    )
        return sorted(assignments, key=lambda row: (-row.fit, row.concept, row.student_id))

    def students_for(self, concept: str) -> list[str]:
        """Student ids best suited to learn ``concept``, highest fit first."""
        ranked = [
            row.student_id
            for row in self.route([concept])
        ]
        seen: list[str] = []
        for student_id in ranked:
            if student_id not in seen:
                seen.append(student_id)
        return seen

    def route_document(self, concepts: Iterable[str]) -> dict[str, list[str]]:
        """Map each concept to the students that should study it."""
        result: dict[str, list[str]] = {}
        for concept in concepts:
            students = self.students_for(concept)
            if students:
                result[concept] = students
        return result
