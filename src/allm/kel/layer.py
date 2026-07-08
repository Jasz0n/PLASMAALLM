"""The Knowledge Evaluation Layer.

Measurement-only (KEL.md section 1): reads the graph, the conflicts
store, teacher state and KDP results; writes nothing except its own
metric time series and graph snapshots (namespaces ``kel_metrics`` /
``kel_snapshots`` — measurements are knowledge about the system and
get the same versioned, append-only treatment as everything else).

Composite score (KEL.md section 6, weights verbatim):

    GHS = 0.25*RCR + 0.15*CRR_norm + 0.20*LG + 0.15*CRE
        + 0.15*GST - 0.10*CD

Normalisation choices (documented because the spec leaves them open):
CRR is unbounded, so it enters as ``min(1, crr / target_reuse)``; LG is
a confidence delta in [-1, 1] and enters raw, so negative learning
genuinely drags the score down. GHS is only computed when every
component is measurable — a composite over missing data would defeat
the "comparability across time" rule.
"""

from __future__ import annotations

import json
from datetime import datetime

from typing import TYPE_CHECKING

from allm.core.logging import get_logger
from allm.kdp.pipeline import DistillationResult
from allm.kel import metrics
from allm.kel.types import Finding, GraphSnapshot, KELConfig, KELReport
from allm.knowledge.graph import KnowledgeGraph
from allm.storage.base import RecordStore
from allm.teacher.state import KnowledgeState

if TYPE_CHECKING:  # avoid a hard dependency; the ledger is optional
    from allm.evidence.ledger import EvidenceLedger

logger = get_logger("kel")

ECOSYSTEM_NAMESPACE = "kel_researcher"
METRICS_NAMESPACE = "kel_metrics"
SNAPSHOT_NAMESPACE = "kel_snapshots"
SNAPSHOT_KEY = "graph"

_GHS_WEIGHTS = {"rcr": 0.25, "crr": 0.15, "lg": 0.20, "cre": 0.15, "gst": 0.15, "cd": -0.10}


class KnowledgeEvaluationLayer:
    """Measures epistemic improvement; never modifies knowledge."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        store: RecordStore,
        state: KnowledgeState,
        config: KELConfig | None = None,
        ledger: "EvidenceLedger | None" = None,
    ) -> None:
        self._graph = graph
        self._store = store
        self._state = state
        self._config = config or KELConfig()
        self._ledger = ledger

    # -- measurement ----------------------------------------------------

    def evaluate(
        self,
        distillation: DistillationResult | None = None,
        *,
        ecosystem: object | None = None,
    ) -> KELReport:
        """Take one full measurement and append it to the time series.

        ``distillation`` feeds RCR when a KDP run just happened;
        otherwise the last recorded RCR carries forward (the graph has
        not been re-distilled, so its redundancy has not changed).

        ``ecosystem`` accepts :class:`ResearcherEcosystemMetrics` from the
        Researcher layer; stored for diagnosis and trend analysis.
        """
        rcr = metrics.rcr(distillation) if distillation is not None else self._last("rcr")
        current_snapshot = metrics.snapshot(self._graph)
        previous_snapshot = self._last_snapshot()
        gst = (
            metrics.stability(previous_snapshot, current_snapshot)
            if previous_snapshot is not None
            else None
        )
        egr = None
        if self._ledger is not None:
            foundation = metrics.evidence_foundation(self._ledger.all_packages())
            egr = metrics.evidence_growth(self._last("evidence_foundation"), foundation)
            self._store.put(
                METRICS_NAMESPACE,
                "evidence_foundation",
                {"value": foundation},
                reason="evidence foundation total (KEL.md 3.7)",
            )
        report = KELReport(
            rcr=rcr,
            cd=metrics.conflict_density(self._graph, self._store),
            gst=gst,
            crr=metrics.concept_reuse(self._graph, self._state),
            lg=metrics.learning_gain(self._graph, self._state),
            cre=metrics.conflict_resolution_efficiency(
                self._graph, self._store, self._state
            ),
            egr=egr,
        )
        report = report.model_copy(update={"ghs": self._ghs(report)})
        self._persist(report, current_snapshot)
        if ecosystem is not None:
            self._persist_ecosystem(ecosystem)
        logger.info(
            "KEL: rcr=%s cd=%s gst=%s crr=%s lg=%s cre=%s egr=%s ghs=%s",
            *(getattr(report, f) for f in ("rcr", "cd", "gst", "crr", "lg", "cre", "egr", "ghs")),
        )
        return report

    def _ghs(self, report: KELReport) -> float | None:
        components = {
            "rcr": report.rcr,
            "crr": None
            if report.crr is None
            else min(1.0, report.crr / self._config.target_reuse),
            "lg": report.lg,
            "cre": report.cre,
            "gst": report.gst,
            "cd": report.cd,
        }
        if any(value is None for value in components.values()):
            return None
        return round(
            sum(_GHS_WEIGHTS[name] * value for name, value in components.items()), 4
        )

    def record_stability(self, ks: float | None) -> None:
        """Persist Knowledge Stability as a first-class KEL metric."""
        self._store.put(
            METRICS_NAMESPACE,
            "ks",
            {"value": ks},
            reason="knowledge stability",
        )
        if ks is not None:
            logger.info("KEL: ks=%.4f", ks)

    # -- time series ------------------------------------------------------

    def history(self, metric: str) -> list[tuple[datetime, float]]:
        """(timestamp, value) series for one metric, oldest first.

        ``None`` measurements are stored (the gap is information) but
        excluded here, since trends are computed over measured values.
        """
        records = self._store.history(METRICS_NAMESPACE, metric)
        return [
            (r.created_at, float(r.value["value"]))
            for r in records
            if r.value["value"] is not None
        ]

    def trend(self, metric: str, *, window: int | None = None) -> float | None:
        """Last minus first over the (windowed) series; None if < 2 points.

        ``window`` maps to KEL.md section 5: short-term = a few
        iterations, long-term = None (whole history).
        """
        series = self.history(metric)
        if window is not None:
            series = series[-window:]
        if len(series) < 2:
            return None
        return round(series[-1][1] - series[0][1], 4)

    # -- failure modes (KEL.md section 9) -----------------------------------

    def diagnose(self) -> list[Finding]:
        """Detect the four failure modes from the latest measurements."""
        cfg = self._config
        findings = []
        rcr, crr, cd = self._last("rcr"), self._last("crr"), self._last("cd")
        cre, gst = self._last("cre"), self._last("gst")
        lg_trend = self.trend("lg")
        ks = self._last("ks")

        if ks is not None and ks < cfg.low_ks:
            findings.append(
                Finding(
                    mode="unstable_mastery",
                    detail=(
                        f"KS {ks:.2f} below {cfg.low_ks:.2f} — "
                        "knowledge is not stable across subsequent learning"
                    ),
                )
            )

        if rcr is not None and crr is not None and rcr > cfg.high_rcr and crr < cfg.low_crr:
            findings.append(
                Finding(
                    mode="false_compression",
                    detail=f"RCR {rcr:.2f} is high but CRR {crr:.2f} is low — "
                    "concepts may be merged incorrectly",
                )
            )
        if crr is not None and crr < cfg.low_crr / 2 and self._graph_grew():
            findings.append(
                Finding(
                    mode="dead_knowledge_growth",
                    detail=f"graph is growing but CRR {crr:.2f} shows concepts "
                    "are never used downstream",
                )
            )
        if cd is not None and cre is not None and cd > cfg.high_cd and cre < cfg.low_cre:
            findings.append(
                Finding(
                    mode="conflict_accumulation",
                    detail=f"CD {cd:.2f} is high while CRE {cre:.2f} is low — "
                    "conflicts pile up without becoming learning",
                )
            )
        if gst is not None and lg_trend is not None and gst > cfg.high_gst and lg_trend < 0:
            findings.append(
                Finding(
                    mode="static_illusion",
                    detail=f"GST {gst:.2f} looks mature but LG trend "
                    f"{lg_trend:+.2f} is declining — stability without learning",
                )
            )
        if self._ledger is not None:
            founded = {p.concept for p in self._ledger.all_packages()}
            unearned = sorted(
                c.name
                for c in self._graph.concepts()
                if c.status == "active"
                and c.confidence > cfg.evidence_confidence_cap
                and c.name not in founded
            )
            if unearned:
                shown = ", ".join(unearned[:3]) + ("…" if len(unearned) > 3 else "")
                findings.append(
                    Finding(
                        mode="unearned_confidence",
                        detail=(
                            f"{len(unearned)} concept(s) above confidence "
                            f"{cfg.evidence_confidence_cap:.2f} with zero evidence "
                            f"packages ({shown}) — documents propose, evidence disposes"
                        ),
                    )
                )
        findings.extend(self._diagnose_ecosystem(cfg))
        return findings

    def _diagnose_ecosystem(self, cfg: KELConfig) -> list[Finding]:
        """Failure modes derived from the latest Researcher ecosystem metrics."""
        metrics = self._last_ecosystem()
        if metrics is None:
            return []

        findings: list[Finding] = []
        if metrics.missing_knowledge >= cfg.high_missing_knowledge:
            findings.append(
                Finding(
                    mode="research_gap",
                    detail=(
                        f"missing knowledge {metrics.missing_knowledge:.2f} — "
                        f"{metrics.recommendation_count} Researcher topics, "
                        f"{metrics.emerging_topics} emerging"
                    ),
                )
            )
        if metrics.research_saturation >= cfg.high_research_saturation:
            findings.append(
                Finding(
                    mode="research_saturation",
                    detail=(
                        f"research saturation {metrics.research_saturation:.2f} — "
                        "recommended topics already mastered"
                    ),
                )
            )
        if metrics.high_conflict_areas >= cfg.high_conflict_discovery:
            findings.append(
                Finding(
                    mode="high_conflict_discovery",
                    detail=(
                        f"high conflict discovery {metrics.high_conflict_areas:.2f} — "
                        f"{metrics.conflict_count} preserved package conflict(s)"
                    ),
                )
            )
        return findings

    # -- internals ------------------------------------------------------------

    def _persist(self, report: KELReport, current: GraphSnapshot) -> None:
        for metric in ("rcr", "cd", "gst", "crr", "lg", "cre", "egr", "ghs", "ks"):
            self._store.put(
                METRICS_NAMESPACE,
                metric,
                {"value": getattr(report, metric)},
                reason="kel evaluation",
            )
        self._store.put(
            SNAPSHOT_NAMESPACE,
            SNAPSHOT_KEY,
            json.loads(current.model_dump_json()),
            reason="kel snapshot",
        )

    def _last(self, metric: str) -> float | None:
        series = self.history(metric)
        return series[-1][1] if series else None

    def _last_snapshot(self) -> GraphSnapshot | None:
        record = self._store.get(SNAPSHOT_NAMESPACE, SNAPSHOT_KEY)
        if record is None:
            return None
        return GraphSnapshot.model_validate(record.value)

    def _graph_grew(self) -> bool:
        history = self._store.history(SNAPSHOT_NAMESPACE, SNAPSHOT_KEY)
        if len(history) < 2:
            return False
        return len(history[-1].value["nodes"]) > len(history[0].value["nodes"])

    def _persist_ecosystem(self, ecosystem) -> None:
        from allm.researcher.ecosystem_metrics import ResearcherEcosystemMetrics

        payload = json.loads(ResearcherEcosystemMetrics.model_validate(ecosystem).model_dump_json())
        self._store.put(
            ECOSYSTEM_NAMESPACE,
            "latest",
            payload,
            reason="researcher ecosystem metrics",
        )

    def _last_ecosystem(self):
        from allm.researcher.ecosystem_metrics import ResearcherEcosystemMetrics

        record = self._store.get(ECOSYSTEM_NAMESPACE, "latest")
        if record is None:
            return None
        return ResearcherEcosystemMetrics.model_validate(record.value)
