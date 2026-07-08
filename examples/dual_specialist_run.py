"""Run one specialist (or generalist) on a domain corpus."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from allm.collector import SamplePool
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, EscalatingGrader, ExactMatchGrader, LLMJudgeGrader, ParaphraseExamGenerator
from allm.kdp.mixed_corpus import contamination_rate, samples_for_identity
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import EchoModel, ModelSpec, load_model
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent, ModelStudentConfig
from allm.students.identity import StudentIdentity
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SpecialistRunResult:
    """Scores and contamination from one specialist loop."""

    student_id: str
    domain: str
    first_score: float
    last_score: float
    peak_score: float
    iterations_completed: int
    samples_studied: int
    contamination: float
    mission_enabled: bool
    workdir: Path


def _student_spec() -> ModelSpec:
    size = os.environ.get("ALLM_STUDENT_SIZE", "large").lower()
    config_name = "ollama_student.yaml" if size == "small" else "ollama_student_7b.yaml"
    default_model = "qwen2.5:0.5b-instruct" if size == "small" else "qwen2.5:7b-instruct"
    model_id = os.environ.get("ALLM_STUDENT_MODEL", default_model)
    spec = ModelSpec.from_yaml(ROOT / "configs/models" / config_name)
    return spec.model_copy(update={"model_id": model_id})


def _grader_spec() -> ModelSpec:
    use_cloud = os.environ.get("ALLM_GRADER", "local").lower() == "cloud"
    name = "ollama_grader_cloud.yaml" if use_cloud else "ollama_grader_local.yaml"
    return ModelSpec.from_yaml(ROOT / "configs/models" / name)


def _make_student(
    student_id: str,
    topic: str,
    train: list[Sample],
    *,
    dry_run: bool,
) -> ModelStudent:
    if dry_run:
        spec = ModelSpec(name=student_id, provider="echo", model_id="none")
        student = ModelStudent(
            student_id,
            topic,
            EchoModel(spec),
            ModelStudentConfig(max_notes=max(len(train), 128), notes_in_prompt=16),
        )
        InContextTrainer().train(student, train)
        return student
    return ModelStudent(
        student_id,
        topic,
        load_model(_student_spec()),
        ModelStudentConfig(max_notes=max(len(train), 64), notes_in_prompt=12),
    )


def run_specialist_loop(
    *,
    student_id: str,
    domain: str,
    train: list[Sample],
    holdout: list[Sample],
    identity: StudentIdentity | None = None,
    dry_run: bool = False,
    workdir: Path | str | None = None,
    verbose: bool = True,
) -> SpecialistRunResult:
    """Run a short learning loop on one domain pool."""
    iterations = int(os.environ.get("ALLM_ITERATIONS", "3"))
    questions = int(os.environ.get("ALLM_QUESTIONS", "6"))
    samples_per_iter = int(os.environ.get("ALLM_SAMPLES_PER_ITER", "16"))
    loop_seed = int(os.environ.get("ALLM_LOOP_SEED", "42"))

    filtered_train = (
        samples_for_identity(train, identity, seed=loop_seed) if identity is not None else train
    )
    if not filtered_train or not holdout:
        raise ValueError(f"{student_id}: empty train ({len(filtered_train)}) or holdout ({len(holdout)})")

    topic = str(filtered_train[0].metadata.get("topic", domain))
    run_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix=f"allm-{student_id}-"))
    store = SQLiteRecordStore(run_dir / "state.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=topic, description=f"{domain} specialist topic"))

    if dry_run:
        grader = ExactMatchGrader("contains")
        teacher = Teacher(
            KnowledgeState(store),
            DatasetExamGenerator(holdout),
            grader,
            TeacherConfig(confidence_smoothing=0.5),
        )
    else:
        judge = LLMJudgeGrader(load_model(_grader_spec()), compare_exact=True)
        grader = EscalatingGrader(ExactMatchGrader("contains"), judge)
        teacher = Teacher(
            KnowledgeState(store),
            ParaphraseExamGenerator(holdout, paraphrase=False),
            grader,
            TeacherConfig(confidence_smoothing=0.5),
        )

    pool = SamplePool()
    pool.ingest(filtered_train)
    student = _make_student(student_id, topic, filtered_train, dry_run=dry_run)
    identities = {student_id: identity} if identity is not None else {}

    loop = LearningLoop(
        teacher=teacher,
        students=[student],
        planner=NeedPlanner(),
        trainer=InContextTrainer(),
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        identities=identities or None,
        enable_peer_consultation=os.environ.get("ALLM_PEER_CONSULT", "0") == "1",
        enable_mediated_consultation=os.environ.get("ALLM_MEDIATED_CONSULT", "0") == "1",
        config=LoopConfig(
            iterations=iterations,
            questions_per_exam=min(questions, len(holdout)),
            samples_per_iteration=min(samples_per_iter, len(filtered_train)),
            study_failures=False,
            seed=loop_seed,
        ),
    )

    if verbose:
        mode = "dry-run" if dry_run else "llm"
        print(f"\n=== {student_id} ({domain}, {mode}) ===")
        print(f"  train={len(filtered_train)} holdout={len(holdout)} mission={'on' if identity else 'off'}")

    first_score = last_score = peak_score = 0.0
    studied_ids: set[str] = set()
    reports = loop.run()
    for report in reports:
        row = report.students[0]
        first_score = first_score or row.score_before
        last_score = row.score_after
        peak_score = max(peak_score, row.score_after)
        studied_ids.update(row.sample_ids)
        if verbose:
            print(f"  iter {report.iteration}: {row.score_before:.2f} -> {row.score_after:.2f} studied={row.samples_studied}")

    contam = contamination_rate(studied_ids, filtered_train, identity, seed=loop_seed) if identity else 0.0
    store.close()
    return SpecialistRunResult(
        student_id=student_id,
        domain=domain,
        first_score=first_score,
        last_score=last_score,
        peak_score=peak_score,
        iterations_completed=len(reports),
        samples_studied=len(studied_ids),
        contamination=contam,
        mission_enabled=identity is not None,
        workdir=run_dir,
    )
