"""Learning strategy profiles — what to study and how to examine."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LearningStrategyName = Literal["definitions", "relations", "reasoning", "research"]


class LearningStrategy(str, Enum):
    """Curriculum phase: definitions → relations → reasoning → research."""

    DEFINITIONS = "definitions"
    RELATIONS = "relations"
    REASONING = "reasoning"
    RESEARCH = "research"


class StrategyProfile(BaseModel):
    """Per-strategy curriculum and exam settings."""

    model_config = ConfigDict(frozen=True)

    sample_kinds: tuple[str, ...]
    use_exam_paraphrase: bool = False
    study_failures: bool = False
    description: str = ""


STRATEGY_PROFILES: dict[LearningStrategy, StrategyProfile] = {
    LearningStrategy.DEFINITIONS: StrategyProfile(
        sample_kinds=("definition", "we_call"),
        description="Terminology and short factual hooks",
    ),
    LearningStrategy.RELATIONS: StrategyProfile(
        sample_kinds=("compact", "teaching"),
        description="Connections and teaching sentences",
    ),
    LearningStrategy.REASONING: StrategyProfile(
        sample_kinds=("compact", "teaching", "definition"),
        use_exam_paraphrase=True,
        description="Paraphrased exams on mixed material",
    ),
    LearningStrategy.RESEARCH: StrategyProfile(
        sample_kinds=("definition", "we_call", "compact", "teaching"),
        use_exam_paraphrase=True,
        study_failures=True,
        description="Evidence-driven learning from failures",
    ),
}

STRATEGY_ORDER: tuple[LearningStrategy, ...] = (
    LearningStrategy.DEFINITIONS,
    LearningStrategy.RELATIONS,
    LearningStrategy.REASONING,
    LearningStrategy.RESEARCH,
)


def profile_for(strategy: LearningStrategy) -> StrategyProfile:
    """Return the curriculum profile for a strategy."""
    return STRATEGY_PROFILES[strategy]


def advance_strategy(current: LearningStrategy) -> LearningStrategy | None:
    """Move to the next strategy phase, or None if already at research."""
    try:
        index = STRATEGY_ORDER.index(current)
    except ValueError:
        return None
    if index + 1 >= len(STRATEGY_ORDER):
        return None
    return STRATEGY_ORDER[index + 1]


def sample_matches_strategy(sample_kind: str | None, kinds: tuple[str, ...]) -> bool:
    """True when a sample kind belongs to the active strategy profile."""
    if not kinds:
        return True
    return (sample_kind or "teaching") in kinds
