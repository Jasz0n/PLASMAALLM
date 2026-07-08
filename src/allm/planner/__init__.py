"""Learning Planner: turns measured state into an ordered roadmap.

Implements Plan.md's curiosity engine
(``Need = Weakness x Importance x Curiosity x Novelty``) with
prerequisite-aware ordering. Signals come from the teacher's knowledge
state plus a topic catalog; the roadmap converts directly into teacher
goals via :meth:`Roadmap.to_goals`.
"""

from allm.planner.base import Planner, planners
from allm.planner.need import NeedPlanner, NeedPlannerConfig
from allm.planner.mission import apply_mission_weights
from allm.planner.router import ConceptAssignment, IngestRouter
from allm.planner.signals import TopicInfo, build_signals, load_catalog
from allm.planner.types import Roadmap, RoadmapItem, TopicSignal

__all__ = [
    "Planner",
    "planners",
    "NeedPlanner",
    "NeedPlannerConfig",
    "apply_mission_weights",
    "ConceptAssignment",
    "IngestRouter",
    "TopicInfo",
    "build_signals",
    "load_catalog",
    "Roadmap",
    "RoadmapItem",
    "TopicSignal",
]
