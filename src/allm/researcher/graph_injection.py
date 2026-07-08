"""Verified packages become graph knowledge (Roadmap M49).

The bridge that lets the planner build a study curriculum for a
repository *without ever seeing raw text*: each package concept lands
in the knowledge graph as a concept with the package as its evidence
entry, append-only like everything else.

Documents propose, evidence disposes (KEL.md 9.5): concepts injected
from a package keep modest confidence — capped below KEL's
unearned-confidence threshold — because a package is a *reading* of
sources, not a reproduced result. Evidence packages raise confidence;
injection never does.
"""

from __future__ import annotations

from allm.core.logging import get_logger
from allm.knowledge.graph import KnowledgeGraph
from allm.knowledge.types import Concept, Evidence
from allm.researcher.types import KnowledgePackage

logger = get_logger("researcher.graph_injection")

# Below KELConfig.evidence_confidence_cap (0.75): package-injected
# concepts can never trip the unearned-confidence detector on arrival.
INJECTION_CONFIDENCE_CAP = 0.6


def inject_package_concepts(
    graph: KnowledgeGraph, package: KnowledgePackage
) -> dict[str, int]:
    """Write a package's concepts into the graph; returns counts.

    New concepts are added with the package as evidence; existing ones
    are revised additively (relations + evidence only — injection never
    rewrites what the graph already believes).
    """
    added = revised = 0
    for concept in package.concepts:
        evidence = Evidence(
            source=package.id,
            detail=f"{package.provider}: {concept.description[:100]}",
            supports=True,
        )
        reason = f"package {package.id} from {package.provider}"
        if graph.get(concept.name) is None:
            graph.add(
                Concept(
                    name=concept.name,
                    description=concept.description,
                    related=concept.relationships,
                    confidence=min(concept.confidence, INJECTION_CONFIDENCE_CAP),
                    # KDP stability differentiates the curriculum: concepts
                    # the corpus states consistently are worth more study
                    # time than one-off mentions.
                    usefulness=round(0.4 + 0.4 * concept.confidence, 4),
                    curiosity=0.7 if concept.knowledge_tier == "hypothesis" else 0.5,
                    evidence=(evidence,),
                    source=package.provider,
                ),
                reason=reason,
            )
            added += 1
        else:
            graph.revise(
                concept.name,
                reason=reason,
                add_related=concept.relationships,
                add_evidence=[evidence],
            )
            revised += 1
    logger.info(
        "injected package %s: %d added, %d revised", package.id, added, revised
    )
    return {"added": added, "revised": revised}
