"""Capability registry and default pipeline order."""

from __future__ import annotations

from allm.researcher.capabilities.book_images import BookImagesCapability
from allm.researcher.capabilities.cross_source import CrossSourceVerificationCapability
from allm.researcher.capabilities.curiosity import ObserveCuriosityCapability
from allm.researcher.capabilities.curriculum import CurriculumTargetingCapability
from allm.researcher.capabilities.curriculum_diagnostics import CurriculumDiagnosticsCapability
from allm.researcher.capabilities.discovery import (
    BookDiscoveryCapability,
    RepositoryDiscoveryCapability,
    SoftwareDiscoveryCapability,
    WorkshopDiscoveryCapability,
)
from allm.researcher.capabilities.ecosystem import EcosystemAnalysisCapability
from allm.researcher.capabilities.economy import EconomyLedgerCapability
from allm.researcher.capabilities.gap_analysis import (
    GraphGapAnalysisCapability,
    MissionReviewCapability,
)
from allm.researcher.capabilities.improvement import ImprovementCapability
from allm.researcher.capabilities.multimodal import (
    MultimodalSyncCapability,
    VideoDiscoveryCapability,
)
from allm.researcher.capabilities.vision import VisionEnrichmentCapability
from allm.researcher.capabilities.audio import AudioEnrichmentCapability
from allm.researcher.capabilities.ocr import OcrEnrichmentCapability
from allm.researcher.capabilities.vision_analytics import VisionAnalyticsCapability
from allm.researcher.capabilities.motion_tracking import MotionTrackingCapability
from allm.researcher.capabilities.motion_continuity import MotionContinuityCapability
from allm.researcher.capabilities.object_identity import ObjectIdentityCapability
from allm.researcher.capabilities.visual_distillation import VisualDistillationCapability
from allm.researcher.capabilities.visual_export import VisualExportCapability
from allm.researcher.capabilities.livekit import (
    LiveKitArchiveCapability,
    LiveKitDiscoveryCapability,
    LiveStreamObserveCapability,
)
from allm.researcher.capabilities.planning import PlanCapability
from allm.researcher.capabilities.understanding import PackageUnderstandingCapability
from allm.researcher.capabilities.verification import GraphVerificationCapability

DEFAULT_PIPELINE: tuple[str, ...] = (
    "observe.curiosity",
    "analysis.gap",
    "missions.review",
    "planning.research",
    "discovery.workshop",
    "discovery.book",
    "discovery.video",
    "discovery.livekit",
    "discovery.software",
    "discovery.repository",
    "understanding.package",
    "understanding.book.images",
    "understanding.sync",
    "understanding.livestream",
    "understanding.vision",
    "understanding.audio",
    "understanding.ocr",
    "understanding.vision.analytics",
    "understanding.vision.motion",
    "understanding.vision.continuity",
    "understanding.vision.identity",
    "understanding.livekit.archive",
    "understanding.visual.distill",
    "verification.graph",
    "verification.cross_source",
    "understanding.visual.export",
    "diagnostics.curriculum",
    "curriculum.target",
    "ecosystem.analyze",
    "economy.ledger",
    "improvement.reflect",
)

_CAPABILITIES = {
    "observe.curiosity": ObserveCuriosityCapability(),
    "analysis.gap": GraphGapAnalysisCapability(),
    "missions.review": MissionReviewCapability(),
    "planning.research": PlanCapability(),
    "discovery.workshop": WorkshopDiscoveryCapability(),
    "discovery.book": BookDiscoveryCapability(),
    "discovery.video": VideoDiscoveryCapability(),
    "discovery.livekit": LiveKitDiscoveryCapability(),
    "discovery.software": SoftwareDiscoveryCapability(),
    "discovery.repository": RepositoryDiscoveryCapability(),
    "understanding.package": PackageUnderstandingCapability(),
    "understanding.book.images": BookImagesCapability(),
    "understanding.sync": MultimodalSyncCapability(),
    "understanding.livestream": LiveStreamObserveCapability(),
    "understanding.vision": VisionEnrichmentCapability(),
    "understanding.audio": AudioEnrichmentCapability(),
    "understanding.ocr": OcrEnrichmentCapability(),
    "understanding.vision.analytics": VisionAnalyticsCapability(),
    "understanding.vision.motion": MotionTrackingCapability(),
    "understanding.vision.continuity": MotionContinuityCapability(),
    "understanding.vision.identity": ObjectIdentityCapability(),
    "understanding.livekit.archive": LiveKitArchiveCapability(),
    "understanding.visual.distill": VisualDistillationCapability(),
    "verification.graph": GraphVerificationCapability(),
    "verification.cross_source": CrossSourceVerificationCapability(),
    "understanding.visual.export": VisualExportCapability(),
    "diagnostics.curriculum": CurriculumDiagnosticsCapability(),
    "curriculum.target": CurriculumTargetingCapability(),
    "ecosystem.analyze": EcosystemAnalysisCapability(),
    "economy.ledger": EconomyLedgerCapability(),
    "improvement.reflect": ImprovementCapability(),
}


def get_capability(name: str):
    """Return a registered capability by name."""
    if name not in _CAPABILITIES:
        raise KeyError(f"unknown capability: {name}")
    return _CAPABILITIES[name]


def pipeline_order(
    enabled: tuple[str, ...] | None = None,
    *,
    discovery_order: str | None = None,
) -> tuple[str, ...]:
    """Resolve capability execution order."""
    if enabled is None:
        order = DEFAULT_PIPELINE
    else:
        order = tuple(name for name in DEFAULT_PIPELINE if name in enabled)
    if discovery_order == "books_first":
        return _swap_discovery(order, "discovery.book", "discovery.workshop")
    if discovery_order == "workshops_first":
        return _swap_discovery(order, "discovery.workshop", "discovery.book")
    return order


def _swap_discovery(order: tuple[str, ...], first: str, second: str) -> tuple[str, ...]:
    """Place ``first`` discovery capability before ``second``."""
    if first not in order or second not in order:
        return order
    rows = list(order)
    i_first, i_second = rows.index(first), rows.index(second)
    if i_first < i_second:
        return order
    rows.remove(first)
    rows.insert(rows.index(second), first)
    return tuple(rows)
