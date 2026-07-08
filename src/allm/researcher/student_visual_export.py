"""Attach Teacher-approved student visual packages to knowledge packages."""

from __future__ import annotations

from allm.researcher.multimodal_types import StudentVisualPackage
from allm.researcher.types import KnowledgePackage


def attach_student_visual_packages(
    package: KnowledgePackage,
    exports: tuple[StudentVisualPackage, ...],
) -> KnowledgePackage:
    """Attach student-safe visual packages to one knowledge package."""
    if not exports:
        return package
    merged = tuple(dict.fromkeys(package.student_visual_packages + exports))
    return package.model_copy(update={"student_visual_packages": merged})
