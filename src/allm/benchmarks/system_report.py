"""State-of-the-system benchmark (Roadmap M47).

One command measures the whole system on the standard corpora and
reports through KEL — the numbers the README quotes and the platform
publishes. Offline by default (echo student, deterministic); a real
Ollama student is opt-in, never a test dependency.

Per corpus: build samples, split train/held-out deterministically,
run the real ``LearningLoop`` (single student, in-context trainer),
take a KEL measurement, then sit the student a held-out exam it never
studied. The held-out gap is reported, not hidden: an echo student
cannot generalize, and the report should say so.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.data.base import Sample

logger = get_logger("benchmarks")

STANDARD_CORPORA = ("fiction", "kids", "books", "practice")

KIDS_SAMPLES_FILE = Path("transcripts/Kids/samples_exam.jsonl")
BOOKS_DIR = Path("books")

# Deterministic fiction micro-domain: facts that cannot exist in any
# pretraining corpus, so every correct answer is *earned* in the loop.
# Superset of the examples/12 facts (same universe).
FICTION_FACTS: dict[str, tuple[str, str]] = {
    "What is the code name of Project Zeta?": ("Nebula", "zeta-program"),
    "How many moons orbit Planet Xerion?": ("7", "xerion-system"),
    "What element powers the Stellar Gate?": ("Quantium", "zeta-program"),
    "Who founded the Lunar Archive?": ("Dr. Venn", "lunar-archive"),
    "What year was the Lunar Archive founded?": ("2214", "lunar-archive"),
    "What color is refined Quantium?": ("violet", "zeta-program"),
    "What is the capital city of Planet Xerion?": ("Vel Kara", "xerion-system"),
    "Who commands the Nebula flagship?": ("Captain Ryx", "zeta-program"),
    "What gas fills Xerion's upper atmosphere?": ("argonelle", "xerion-system"),
    "How many wings does the Lunar Archive have?": ("5", "lunar-archive"),
    "What is the Stellar Gate's transit time to Xerion?": ("9 hours", "zeta-program"),
    "Who catalogued the first argonelle cloud?": ("Surveyor Ila", "xerion-system"),
}


class CorpusReport(BaseModel):
    """Benchmark numbers for one corpus."""

    model_config = ConfigDict(frozen=True)

    corpus: str
    samples_train: int
    samples_holdout: int
    iterations: int
    score_before: float
    score_after: float
    holdout_score: float
    holdout_gap: float
    learning_gain: float | None
    ghs: float | None
    rcr: float | None
    conflict_density: float | None
    concept_reuse: float | None
    conflict_resolution: float | None
    evidence_growth: float | None = None  # EGR: did the run earn evidence?


class SystemReport(BaseModel):
    """The state-of-the-system report (Roadmap M47)."""

    model_config = ConfigDict(frozen=True)

    created_at: str
    student_provider: str
    seed: int
    iterations: int
    corpora: tuple[CorpusReport, ...]
    skipped: tuple[str, ...] = ()  # corpora unavailable on this checkout

    def to_markdown(self) -> str:
        """Render the report as a README-ready markdown table."""
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        lines = [
            f"State of the system — student `{self.student_provider}`, "
            f"seed {self.seed}, {self.iterations} iteration(s), {self.created_at}",
            "",
            "| Corpus | Train | Held-out | Before | After | Held-out score | Gap | LG | EGR | GHS | RCR |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in self.corpora:
            lines.append(
                f"| {row.corpus} | {row.samples_train} | {row.samples_holdout} "
                f"| {row.score_before:.2f} | {row.score_after:.2f} "
                f"| {row.holdout_score:.2f} | {row.holdout_gap:.2f} "
                f"| {fmt(row.learning_gain)} | {fmt(row.evidence_growth)} "
                f"| {fmt(row.ghs)} | {fmt(row.rcr)} |"
            )
        for note in self.skipped:
            lines.append(f"\nskipped: {note}")
        return "\n".join(lines)


def fiction_samples() -> list[Sample]:
    """The deterministic fiction micro-domain."""
    return [
        Sample(id=f"fiction-{i:02d}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(sorted(FICTION_FACTS.items()))
    ]


def kids_samples(root: Path, limit: int = 24) -> list[Sample]:
    """Deterministic slice of the exam-friendly kids workshop corpus."""
    path = root / KIDS_SAMPLES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"kids corpus not found: {path} (run examples/15_kids_corpus_graph.py)"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["id"])
    samples = [
        Sample(
            id=row["id"],
            input=row["input"],
            target=row["target"],
            metadata={
                "topic": row.get("topic", "kids-plasma"),
                "sample_kind": row.get("sample_kind", "teaching"),
                "source": row.get("source", ""),
            },
        )
        for row in rows
    ]
    return samples[:limit] if limit else samples


def book_corpus(root: Path, store, graph, *, limit: int = 24):
    """Distill the book sidecar texts and derive exam samples.

    Returns ``(samples, DistillationResult)``; the distillation is
    injected into ``graph`` so KEL sees the real structure and RCR.
    """
    from allm.kdp import DocumentStore, GraphInjector, KDPipeline

    books_dir = root / BOOKS_DIR
    files = sorted(books_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no book sidecar texts under {books_dir}")
    documents = DocumentStore(None)
    for file in files:
        documents.ingest_file(file)
    result = KDPipeline().distill(documents)
    GraphInjector(graph, store).inject(result)

    samples = []
    for unit in result.units:
        if unit.type not in ("concept", "fact"):
            continue
        samples.append(
            Sample(
                id=f"book-{unit.id}",
                input=f"What is {unit.normalized_concept}?",
                target=unit.content,
                metadata={"topic": unit.normalized_concept},
            )
        )
    samples.sort(key=lambda s: s.id)
    return samples[:limit] if limit else samples, result


def _practice_procedures():
    from allm.practice import PracticeProcedure, VariableSpec

    return (
        PracticeProcedure(
            id="collatz_steps",
            description="Count how many Collatz iterations a starting integer needs to reach 1.",
            program=(
                "steps = 0\n"
                "n = start\n"
                "while n != 1:\n"
                "    n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
                "    steps += 1\n"
                "print(steps)\n"
            ),
            variables=(VariableSpec(name="start", default=6, candidates=(7, 9, 27)),),
            topic="number-theory",
        ),
        PracticeProcedure(
            id="compound_growth",
            description="Compound an amount over periods at a fixed rate; print the result rounded to 2 places.",
            program="print(round(amount * (1 + rate) ** periods, 2))\n",
            variables=(
                VariableSpec(name="rate", default=0.1, candidates=(0.25, 0.5)),
                VariableSpec(name="periods", default=3, candidates=(10,)),
            ),
            topic="finance",
        ),
    )


def practice_corpus(graph, ledger) -> list[Sample]:
    """Earn a corpus by doing: sweeps → evidence packages → samples.

    Unlike the text corpora, every sample here is backed by a captured
    execution and every run is in the ledger — the benchmark's EGR
    column measures exactly this difference.
    """
    from allm.practice import (
        SandboxExecutor,
        practice_samples,
        record_sweep,
        run_sweep,
        run_to_package,
    )

    executor = SandboxExecutor()
    samples: list[Sample] = []
    for procedure in _practice_procedures():
        runs = []
        for spec in procedure.variables:
            sweep = run_sweep(procedure, spec.name, executor=executor)
            record_sweep(graph, procedure, sweep)
            runs.extend(sweep.runs)
        unique = list({run.id: run for run in runs}.values())
        for run in unique:
            ledger.submit(run_to_package(procedure, run))
        samples.extend(practice_samples(procedure, tuple(unique)))
    return sorted(samples, key=lambda s: s.id)


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """Deterministic ~80/20 train/held-out split (every 5th held out)."""
    if len(samples) < 2:
        raise ValueError("need at least two samples to hold one out")
    ordered = sorted(samples, key=lambda s: s.id)
    holdout = ordered[4::5] or [ordered[-1]]
    held_ids = {s.id for s in holdout}
    train = [s for s in ordered if s.id not in held_ids]
    return train, holdout


def _make_student(provider: str):
    from allm.models import ModelSpec
    from allm.students import ModelStudent

    if provider == "echo":
        from allm.models import EchoModel

        spec = ModelSpec(name="bench", provider="echo", model_id="none")
        return ModelStudent("bench", "general", EchoModel(spec))
    if provider == "ollama":
        import os

        from allm.models.base import model_loaders

        model_id = os.environ.get("ALLM_STUDENT_MODEL", "qwen2.5:7b-instruct")
        spec = ModelSpec(name="bench", provider="ollama", model_id=model_id)
        return ModelStudent("bench", "general", model_loaders.get("ollama")().load(spec))
    raise ValueError(f"unknown student provider {provider!r} (echo, ollama)")


def _run_corpus(
    corpus: str,
    *,
    root: Path,
    workdir: Path,
    iterations: int,
    seed: int,
    limit: int,
    student_provider: str,
) -> CorpusReport:
    from allm.collector import SamplePool
    from allm.evidence import EvidenceLedger
    from allm.exam import DatasetExamGenerator, ExactMatchGrader
    from allm.kel import KnowledgeEvaluationLayer
    from allm.knowledge import Concept, KnowledgeGraph
    from allm.loop import LearningLoop, LoopConfig
    from allm.memory import EpisodicMemory
    from allm.planner import NeedPlanner
    from allm.storage import SQLiteRecordStore
    from allm.students import FailureLog
    from allm.teacher import KnowledgeState, Teacher, TeacherConfig
    from allm.trainer import InContextTrainer

    store = SQLiteRecordStore(workdir / f"{corpus}.sqlite3")
    try:
        graph = KnowledgeGraph(store)
        ledger = EvidenceLedger(store)
        state = KnowledgeState(store)
        kel = KnowledgeEvaluationLayer(graph, store, state, ledger=ledger)
        # Baseline measurement before anything is learned or earned, so
        # the final EGR reflects evidence produced *by this run*.
        kel.evaluate()
        distillation = None
        if corpus == "fiction":
            samples = fiction_samples()
        elif corpus == "kids":
            samples = kids_samples(root, limit)
        elif corpus == "books":
            samples, distillation = book_corpus(root, store, graph, limit=limit)
        elif corpus == "practice":
            samples = practice_corpus(graph, ledger)
            samples = samples[:limit] if limit else samples
        else:
            raise ValueError(f"unknown corpus {corpus!r} (choose from {STANDARD_CORPORA})")

        train, holdout = split_samples(samples)

        # KEL's learning gain only counts topics that exist as graph
        # concepts; seed any topic the distillation did not already add.
        existing = {c.name for c in graph.concepts()}
        for topic in sorted({str(s.metadata.get("topic", "general")) for s in train}):
            if topic not in existing:
                graph.add(Concept(name=topic, usefulness=0.8))

        grader = ExactMatchGrader()
        teacher = Teacher(
            state,
            DatasetExamGenerator(train),
            grader,
            TeacherConfig(confidence_smoothing=1.0),
        )
        pool = SamplePool()
        pool.ingest(train)
        student = _make_student(student_provider)
        loop = LearningLoop(
            teacher=teacher,
            students=[student],
            planner=NeedPlanner(),
            trainer=InContextTrainer(),
            pool=pool,
            memory=EpisodicMemory(store),
            failure_log=FailureLog(store),
            graph=graph,
            config=LoopConfig(
                iterations=iterations,
                questions_per_exam=min(6, len(train)),
                samples_per_iteration=max(16, len(train)),
                seed=seed,
            ),
        )
        reports = loop.run()

        # Measure before the held-out exam so the KEL numbers describe
        # the training run, not the held-out sitting. EGR compares
        # against the baseline measurement taken before the run.
        report = kel.evaluate(distillation)

        holdout_exam = DatasetExamGenerator(holdout).generate(
            num_questions=len(holdout), seed=seed + 500
        )
        graded = [grader.grade(q, student.solve(q)) for q in holdout_exam.questions]
        holdout_score = sum(1 for g in graded if g.correct) / len(graded)

        score_before = reports[0].students[0].score_before
        score_after = reports[-1].students[0].score_after
        logger.info(
            "%s: %d train / %d held-out, %.2f -> %.2f, held-out %.2f",
            corpus, len(train), len(holdout), score_before, score_after, holdout_score,
        )
        return CorpusReport(
            corpus=corpus,
            samples_train=len(train),
            samples_holdout=len(holdout),
            iterations=iterations,
            score_before=score_before,
            score_after=score_after,
            holdout_score=holdout_score,
            holdout_gap=round(score_after - holdout_score, 4),
            learning_gain=report.lg,
            ghs=report.ghs,
            rcr=report.rcr,
            conflict_density=report.cd,
            concept_reuse=report.crr,
            conflict_resolution=report.cre,
            evidence_growth=report.egr,
        )
    finally:
        store.close()


def run_system_benchmark(
    corpora: tuple[str, ...] = STANDARD_CORPORA,
    *,
    iterations: int = 3,
    seed: int = 13,
    limit: int = 24,
    student: str = "echo",
    root: Path | None = None,
    workdir: Path | None = None,
) -> SystemReport:
    """Run the full benchmark and return the report."""
    base = root or Path(__file__).resolve().parents[3]
    rows = []
    skipped = []
    with tempfile.TemporaryDirectory(prefix="allm-benchmark-") as tmp:
        work = workdir or Path(tmp)
        for corpus in corpora:
            try:
                rows.append(
                    _run_corpus(
                        corpus,
                        root=base,
                        workdir=work,
                        iterations=iterations,
                        seed=seed,
                        limit=limit,
                        student_provider=student,
                    )
                )
            except FileNotFoundError as exc:
                # A missing local corpus (e.g. the uncommitted books
                # directory) must not sink the whole report.
                skipped.append(f"{corpus} — {exc}")
                logger.warning("corpus %s skipped: %s", corpus, exc)
    return SystemReport(
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        student_provider=student,
        seed=seed,
        iterations=iterations,
        corpora=tuple(rows),
        skipped=tuple(skipped),
    )
