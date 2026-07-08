"""Continuous Learning Loop: measure -> plan -> collect -> learn ->
debate -> test -> compress -> update memory -> repeat.

The loop is the composition root over all other packages; it owns the
order of calls, not the logic.
"""

from allm.loop.history import IterationHistoryWriter, LearningIterationRecord, LearningRunManifest
from allm.loop.kel_steered_loop import KelSteeredLearningLoop
from allm.loop.kel_steering import KelSteeringConfig, KelSteeringDecision, KelSteeringPolicy
from allm.loop.strategy import LearningStrategy, StrategyProfile, profile_for
from allm.loop.learning_loop import (
    IterationReport,
    LearningLoop,
    LoopConfig,
    StudentIteration,
)

__all__ = [
    "IterationReport",
    "IterationHistoryWriter",
    "KelSteeredLearningLoop",
    "KelSteeringConfig",
    "KelSteeringDecision",
    "KelSteeringPolicy",
    "LearningIterationRecord",
    "LearningRunManifest",
    "LearningLoop",
    "LearningStrategy",
    "LoopConfig",
    "StrategyProfile",
    "StudentIteration",
    "profile_for",
]
