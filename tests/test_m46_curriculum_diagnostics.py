"""Tests for M46 Researcher model router and curriculum diagnostics."""

from __future__ import annotations

from allm.kel.research_requests import KelResearchRequest
from allm.loop.history import LearningIterationRecord
from allm.models.base import ModelSpec
from allm.models.echo import EchoModel
from allm.researcher.curriculum_diagnostics import (
    DiagnosticContext,
    build_reasoning_prompt,
    diagnose_heuristic,
    diagnose_with_reasoning,
    parse_reasoning_response,
)
from allm.researcher.layer import ResearcherLayer
from allm.researcher.model_router import route_request, resolve_model_spec
from allm.researcher.remediation import requests_to_recommendations
from allm.storage import SQLiteRecordStore


def _request(trigger: str = "strategy_stagnation") -> KelResearchRequest:
    return KelResearchRequest(
        id="req-1",
        topic="kids-plasma",
        task="Improve relation learning",
        trigger=trigger,  # type: ignore[arg-type]
        priority=0.9,
        student_id="kids-kel",
        search_hints=("relations", "prerequisite"),
        reason="relations stagnation",
    )


def _history() -> tuple[LearningIterationRecord, ...]:
    return (
        LearningIterationRecord(
            iteration=1,
            strategy="definitions",
            sample_kinds=("definition",),
            sample_ids=("s1",),
            student_id="kids-kel",
            score_before=0.0,
            score_after=0.25,
            goals=("kids-plasma",),
            samples_studied=20,
            questions_per_exam=8,
            use_exam_paraphrase=False,
            kel_lg=0.07,
            metadata={"kel_ks": 0.52},
        ),
        LearningIterationRecord(
            iteration=5,
            strategy="relations",
            sample_kinds=("compact",),
            sample_ids=("s2",),
            student_id="kids-kel",
            score_before=0.0,
            score_after=0.0,
            goals=("kids-plasma",),
            samples_studied=10,
            questions_per_exam=8,
            use_exam_paraphrase=False,
            kel_lg=0.09,
            failure_prompts=(
                "What is Principal Star?",
                "What is Magravs Created By This?",
                "What is Therefore Man?",
            ),
            metadata={"kel_ks": 0.33},
        ),
    )


def test_route_request_prefers_verifier_for_conflicts() -> None:
    request = _request("high_conflict")
    assert route_request(request) == "verifier"


def test_route_request_prefers_reasoning_for_relations() -> None:
    assert route_request(_request()) == "reasoning"


def test_resolve_model_spec_reasoning() -> None:
    spec = resolve_model_spec("reasoning")
    assert spec.provider == "ollama"
    assert "qwen" in spec.model_id


def test_heuristic_diagnoses_weak_relations() -> None:
    diagnostic = diagnose_heuristic(
        _request(),
        DiagnosticContext(
            failure_prompts=_history()[-1].failure_prompts,
            strategy="relations",
            kel_ks=0.33,
            conflict_count=2611,
            history=_history(),
        ),
    )
    assert diagnostic.failure_reason in {
        "weak_relation_material",
        "missing_prerequisite",
        "conflicting_sources",
    }
    assert diagnostic.recommendations


def test_reasoning_parser() -> None:
    raw = (
        "FAILURE_REASON: missing_prerequisite\n"
        "CONFIDENCE: 0.82\n"
        "EVIDENCE: 83% of failed questions need concept X before Y\n"
        "RECOMMENDATION: Create visual prerequisite lesson\n"
    )
    reason, confidence, evidence, recommendations = parse_reasoning_response(raw)
    assert reason == "missing_prerequisite"
    assert confidence == 0.82
    assert evidence[0].startswith("83%")
    assert "visual" in recommendations[0]


def test_reasoning_diagnostic_with_echo_model() -> None:
    spec = ModelSpec(name="reasoning", provider="echo", model_id="none")
    model = EchoModel(spec)
    prompt = build_reasoning_prompt(
        _request(),
        DiagnosticContext(failure_prompts=("What is Principal Star?",), strategy="relations"),
    )
    model.script(
        prompt,
        (
            "FAILURE_REASON: missing_prerequisite\n"
            "CONFIDENCE: 0.8\n"
            "EVIDENCE: relation questions require undefined terms\n"
            "RECOMMENDATION: add prerequisite definition module\n"
        ),
    )
    diagnostic = diagnose_with_reasoning(
        _request(),
        DiagnosticContext(failure_prompts=("What is Principal Star?",), strategy="relations"),
        model,
        spec=spec,
    )
    assert diagnostic.failure_reason == "missing_prerequisite"
    assert diagnostic.model_id == "none"


def test_remediation_includes_diagnosis() -> None:
    diagnostic = diagnose_heuristic(
        _request("high_conflict"),
        DiagnosticContext(conflict_count=500, history=_history()),
    )
    recs = requests_to_recommendations((_request("high_conflict"),), diagnostics=(diagnostic,))
    assert "Diagnosis:" in recs[0].proposal_hint
    assert "conflicting_sources" in recs[0].proposal_hint


def test_researcher_submit_runs_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_CURRICULUM_DIAGNOSTICS", "1")
    monkeypatch.setenv("ALLM_CURRICULUM_DIAGNOSTICS_BACKEND", "heuristic")
    store = SQLiteRecordStore(":memory:")
    layer = ResearcherLayer(store)
    request = _request("repair_mode")
    count = layer.submit_kel_research_requests(
        (request,),
        diagnostic_context=DiagnosticContext(
            failure_prompts=("What is Principal Star?", "What is Magravs Created By This?"),
            strategy="relations",
            kel_ks=0.32,
            conflict_count=120,
            history=_history(),
        ),
    )
    assert count == 1
    record = store.get("curriculum_diagnostics", request.id)
    assert record is not None
    assert record.value["failure_reason"] in {
        "missing_prerequisite",
        "retention_instability",
        "conflicting_sources",
        "weak_relation_material",
    }
    active = layer.active_recommendations()
    assert "Diagnosis:" in active[0].proposal_hint
