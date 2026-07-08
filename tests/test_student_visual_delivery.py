"""Tests for delivering Teacher-approved visuals to students (M25)."""

from allm.researcher.multimodal_types import DistilledVisualBrief, StudentVisualPackage
from allm.researcher.packages import package_from_workshop_dir
from allm.researcher.student_visual_export import attach_student_visual_packages
from allm.students.model_student import ModelStudent, ModelStudentConfig
from allm.models.echo import EchoModel
from allm.models.base import ModelSpec
from allm.teacher.student_visual_delivery import (
    count_visual_notes_delivered,
    deliver_visual_notes,
    packages_for_topic,
    visual_study_notes,
)
from allm.memory.episodic import EpisodicMemory
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_export import approve_visual_brief, export_approved_briefs
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _export() -> StudentVisualPackage:
    brief = DistilledVisualBrief(
        brief_id="dvis_delivery",
        concept_name="plasma motion",
        concept_description="Magnet rotation without fuel",
        images=("Rotating magnets chasing each other",),
        diagram_summary="North and south pole labels",
        explanations=("The field showed magnetical beat.",),
        experiment_prompt="Observe magnet motion on a pin.",
        questions=("What did the workshop show?",),
        evidence_confidence=0.9,
    )
    return export_approved_briefs(
        (brief,),
        (approve_visual_brief(brief),),
        curriculum_topic="kids-plasma",
    )[0]


def test_visual_study_notes_cover_export_fields() -> None:
    export = _export()
    notes = visual_study_notes(export)
    prompts = [row[0] for row in notes]
    assert any("plasma motion" in prompt for prompt in prompts)
    assert any("diagram" in prompt.lower() for prompt in prompts)
    assert any("experiment" in prompt.lower() for prompt in prompts)


def test_packages_for_topic_filters_by_curriculum() -> None:
    export = _export()
    package = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=1,
        curriculum_topic="kids-plasma",
    )
    updated = attach_student_visual_packages(package, (export,))
    matched = packages_for_topic((updated,), "kids-plasma")
    assert len(matched) == 1
    assert packages_for_topic((updated,), "fastify-api") == ()


def test_deliver_visual_notes_injects_student_memory() -> None:
    student = ModelStudent(
        "kids-kel",
        "kids-plasma",
        EchoModel(ModelSpec(name="echo", provider="echo", model_id="none")),
        ModelStudentConfig(max_notes=32),
    )
    before = len(student.notes)
    delivered = deliver_visual_notes(student, (_export(),))
    assert delivered > 0
    assert len(student.notes) == before + delivered


def test_count_visual_notes_delivered_sums_episodes(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "visual.sqlite3")
    memory = EpisodicMemory(store)
    memory.remember(
        "kids-kel",
        "observation",
        "Teacher delivered 5 visual study note(s)",
        detail={"visual_notes": 5},
    )
    memory.remember(
        "kids-kel",
        "observation",
        "consulted peer on plasma",
        detail={"expert_id": "peer-1"},
    )
    memory.remember(
        "kids-kel",
        "observation",
        "Teacher delivered 3 visual study note(s)",
        detail={"visual_notes": 3},
    )
    assert count_visual_notes_delivered(memory, "kids-kel") == 8
    store.close()
