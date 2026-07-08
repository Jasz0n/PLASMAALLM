"""Tests for learning strategy profiles and KEL strategy switching."""

from allm.kel.layer import KnowledgeEvaluationLayer
from allm.kel.types import KELConfig
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop.history import IterationHistoryWriter, LearningIterationRecord
from allm.loop.kel_steering import KelSteeringConfig, KelSteeringPolicy, apply_steering
from allm.loop.learning_loop import IterationReport, LoopConfig, StudentIteration
from allm.loop.strategy import LearningStrategy, advance_strategy, profile_for
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


def _kel() -> KnowledgeEvaluationLayer:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="topic-a", description="A"))
    return KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), KELConfig(high_gst=0.5))


def _report(score: float) -> IterationReport:
    return IterationReport(
        iteration=1,
        students=(
            StudentIteration(
                student_id="s1",
                score_before=0.0,
                score_after=score,
                goals=("topic-a",),
                samples_studied=4,
                strategy="definitions",
            ),
        ),
        debate_disagreement=None,
        compression_applied=0,
        compression_retracted=0,
    )


def test_advance_strategy_order() -> None:
    assert advance_strategy(LearningStrategy.DEFINITIONS) == LearningStrategy.RELATIONS
    assert advance_strategy(LearningStrategy.RESEARCH) is None


def test_mastery_advances_strategy_on_peak() -> None:
    policy = KelSteeringPolicy(
        KelSteeringConfig(strategy_advance_threshold=0.35, strategy_advance_window=3)
    )
    active = LoopConfig(strategy="definitions", sample_kinds=("definition", "we_call"))
    reports = [_report(0.12), _report(0.23), _report(0.35)]
    decision = policy.decide(4, reports, _kel(), active)
    assert decision.strategy == LearningStrategy.RELATIONS


def test_rolling_average_can_advance_strategy() -> None:
    policy = KelSteeringPolicy(
        KelSteeringConfig(strategy_advance_threshold=0.4, strategy_advance_window=2)
    )
    active = LoopConfig(strategy="definitions", sample_kinds=("definition", "we_call"))
    reports = [_report(0.38), _report(0.42)]
    decision = policy.decide(3, reports, _kel(), active)
    assert decision.strategy == LearningStrategy.RELATIONS


def test_apply_steering_sets_profile_fields() -> None:
    from allm.loop.kel_steering import KelSteeringDecision

    active = LoopConfig()
    decision = KelSteeringDecision(strategy=LearningStrategy.REASONING)
    updated = apply_steering(active, decision)
    profile = profile_for(LearningStrategy.REASONING)
    assert updated.sample_kinds == profile.sample_kinds
    assert updated.use_exam_paraphrase is True


def test_history_writer_roundtrip(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    writer = IterationHistoryWriter(path)
    record = LearningIterationRecord(
        iteration=1,
        strategy="definitions",
        sample_kinds=("definition",),
        sample_ids=("s1",),
        student_id="kid",
        score_before=0.0,
        score_after=0.25,
        goals=("kids-plasma",),
        samples_studied=8,
        questions_per_exam=4,
        use_exam_paraphrase=False,
        kel_lg=0.1,
        holdout_count=212,
        holdout_answers_in_train=0,
        holdout_novel_lexical=0,
    )
    writer.append(record)
    loaded = writer.load_all()
    assert len(loaded) == 1
    assert loaded[0].holdout_answers_in_train == 0


def test_manifest_written(tmp_path) -> None:
    from allm.loop.history import LearningRunManifest

    writer = IterationHistoryWriter(tmp_path / "history.jsonl")
    manifest = LearningRunManifest(
        student_model="qwen2.5:7b-instruct",
        train_count=572,
        holdout_count=212,
        holdout_exact_prompt_matches=0,
        holdout_high_overlap=176,
        holdout_low_overlap=36,
        holdout_novel_lexical=0,
        holdout_answers_in_train=0,
        kel_mastery_threshold=0.75,
        kel_strategy_advance_threshold=0.4,
    )
    manifest_path = writer.write_manifest(manifest)
    assert manifest_path.is_file()
    assert manifest.holdout_answers_in_train == 0
