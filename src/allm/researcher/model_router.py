"""Researcher model router — Chief Scientist orchestrates specialist models (M46)."""

from __future__ import annotations

import os
from typing import Literal

from allm.kel.research_requests import KelResearchRequest, TriggerKind
from allm.models.base import ModelSpec

SpecialistRole = Literal["reasoning", "verifier", "vision", "efficient"]

_VISION_HINTS = frozenset({"visual", "diagram", "worked_example"})
_VERIFIER_HINTS = frozenset({"conflict_resolution", "cross_source", "terminology"})
_REASONING_HINTS = frozenset(
    {"prerequisite", "relations", "examples", "application", "experiment", "discovery"}
)

_TRIGGER_DEFAULTS: dict[TriggerKind, SpecialistRole] = {
    "high_conflict": "verifier",
    "strategy_stagnation": "reasoning",
    "research_gap": "reasoning",
    "forgetting": "reasoning",
    "repair_mode": "reasoning",
    "unstable_mastery": "reasoning",
}


def model_router_enabled() -> bool:
    """Return True when Researcher should route tasks to specialist models."""
    return os.environ.get("ALLM_RESEARCHER_MODEL_ROUTER", "1") == "1"


def route_request(request: KelResearchRequest) -> SpecialistRole:
    """Pick the specialist role best suited to one KEL research request."""
    hints = {hint.lower() for hint in request.search_hints}
    if hints & _VERIFIER_HINTS or request.trigger == "high_conflict":
        return "verifier"
    if hints & _VISION_HINTS and request.trigger in {"repair_mode", "unstable_mastery"}:
        return "vision"
    if hints & _REASONING_HINTS:
        return "reasoning"
    return _TRIGGER_DEFAULTS.get(request.trigger, "reasoning")


def resolve_model_spec(role: SpecialistRole) -> ModelSpec:
    """Resolve a specialist role to a concrete :class:`ModelSpec`."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    defaults: dict[SpecialistRole, tuple[str, str]] = {
        "reasoning": (
            "ALLM_RESEARCHER_REASONING_MODEL",
            "qwen2.5:14b-instruct",
        ),
        "verifier": (
            "ALLM_RESEARCHER_VERIFIER_MODEL",
            "qwen2.5:14b-instruct",
        ),
        "vision": (
            "ALLM_RESEARCHER_VISION_MODEL",
            os.environ.get("ALLM_VISION_MODEL", "llava"),
        ),
        "efficient": (
            "ALLM_RESEARCHER_EFFICIENT_MODEL",
            "qwen2.5:7b-instruct",
        ),
    }
    env_key, default_id = defaults[role]
    model_id = os.environ.get(env_key, default_id)
    if role == "vision":
        return ModelSpec(name=f"researcher-{role}", provider="ollama", model_id=model_id)
    config_name = {
        "reasoning": "ollama_digest_writer.yaml",
        "verifier": "ollama_grader_local.yaml",
        "efficient": "ollama_student_7b.yaml",
    }.get(role, "ollama_grader_local.yaml")
    spec = ModelSpec.from_yaml(root / "configs" / "models" / config_name)
    return spec.model_copy(update={"model_id": model_id, "name": f"researcher-{role}"})


def describe_route(request: KelResearchRequest) -> str:
    """Human-readable routing summary for logs."""
    role = route_request(request)
    spec = resolve_model_spec(role)
    return f"{role}::{spec.model_id}"
