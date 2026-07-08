"""Evidential confidence: reproducible results, not popularity.

Design decisions
----------------
- Kind weights encode "how much can this move a belief": independent
  replications are the strongest signal, hands-on experiments next,
  papers/transcripts/debates are context, observations the weakest.
- Popularity resistance: per (contributor, stance) only the single
  strongest package counts. One lab posting the same success fifty
  times moves confidence exactly as much as posting it once; a second
  *independent* contributor moves it a lot.
- A replication only earns the replication weight if it re-runs
  another contributor's package (``replicates`` set, different
  contributor); replicating yourself counts as a plain experiment.
- Confidence = Laplace-smoothed support share:
      (support + 1) / (support + challenge + inconclusive/2 + 2)
  One unchallenged experiment lands near 0.6, not 1.0 — certainty has
  to be earned by independent weight. Inconclusive results are real
  uncertainty, so they enter the denominator at half weight.
"""

from __future__ import annotations

from typing import Sequence

from allm.evidence.types import ConfidenceBreakdown, EvidencePackage

KIND_WEIGHTS: dict[str, float] = {
    "replication": 1.0,
    "experiment": 0.8,
    "paper": 0.4,
    "debate": 0.4,
    "transcript": 0.3,
    "observation": 0.3,
}


def package_weight(package: EvidencePackage, by_id: dict[str, EvidencePackage]) -> float:
    """Effective weight of one package (before popularity capping)."""
    if package.kind == "replication":
        original = by_id.get(package.replicates or "")
        independent = original is not None and original.contributor != package.contributor
        return KIND_WEIGHTS["replication" if independent else "experiment"]
    return KIND_WEIGHTS[package.kind]


def evidential_confidence(
    concept: str, packages: Sequence[EvidencePackage]
) -> ConfidenceBreakdown | None:
    """Compute confidence for ``concept`` from its packages.

    ``None`` when there are no packages — unmeasured belief must never
    look like measured 0.5.
    """
    relevant = [p for p in packages if p.concept == concept]
    if not relevant:
        return None
    by_id = {p.id: p for p in relevant}

    # Popularity resistance: strongest package per (contributor, stance).
    strongest: dict[tuple[str, str], float] = {}
    for package in relevant:
        key = (package.contributor, package.outcome)
        weight = package_weight(package, by_id)
        strongest[key] = max(strongest.get(key, 0.0), weight)

    support = sum(w for (_, stance), w in strongest.items() if stance == "supported")
    challenge = sum(w for (_, stance), w in strongest.items() if stance == "challenged")
    inconclusive = sum(
        w for (_, stance), w in strongest.items() if stance == "inconclusive"
    )
    value = (support + 1.0) / (support + challenge + inconclusive / 2.0 + 2.0)

    replications = sum(
        1
        for p in relevant
        if p.kind == "replication"
        and p.replicates in by_id
        and by_id[p.replicates].contributor != p.contributor
    )
    return ConfidenceBreakdown(
        concept=concept,
        value=round(value, 4),
        support_weight=round(support, 4),
        challenge_weight=round(challenge, 4),
        inconclusive_weight=round(inconclusive, 4),
        contributors=len({p.contributor for p in relevant}),
        independent_replications=replications,
        packages=tuple(sorted(by_id)),
    )
