"""smallVision.md demo: the human-AI loop over evidence packages.

Conflicting workshop transcripts flow through KDP into the graph; the
contradiction becomes an experiment proposal; humans claim it and
resolve it with evidence packages (an experiment, an independent
replication, a failed replication); confidence updates from
reproducible results — and the full provenance tree shows why.

    python examples/09_evidence_and_proposals.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.evidence import Artifact, EvidenceBinder, EvidenceLedger, EvidencePackage
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard
from allm.storage import SQLiteRecordStore

WORKSHOP_1 = """The Peltier Stack is a converter that exceeds 80 percent
efficiency in bench tests.
"""

WORKSHOP_2 = """The Peltier Stack is an ordinary thermoelectric module capped
near 10 percent efficiency by material physics.
"""


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-evidence-"))
    store = SQLiteRecordStore(workdir / "allm.sqlite3")
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    binder = EvidenceBinder(graph, ledger)
    board = ProposalBoard(store, binder)

    print("=== 1. Community discussion arrives (KDP) ===")
    documents = DocumentStore(store)
    documents.ingest_text("workshop-1.md", WORKSHOP_1, context="energy")
    documents.ingest_text("workshop-2.md", WORKSHOP_2, context="energy")
    result = KDPipeline().distill(documents)
    GraphInjector(graph, store).inject(result)
    conflict = result.conflicts[0]
    print(f"  contradiction detected on {conflict.concept!r}")

    print("\n=== 2. The contradiction becomes an experiment proposal ===")
    proposal = board.from_conflict(conflict)
    print(f"  {proposal.id} [{proposal.status}] {proposal.question}")
    print(f"  rationale: {proposal.rationale[:100]}...")

    print("\n=== 3. A human claims it and runs the experiment ===")
    board.claim(proposal.id, "alice-lab")
    original = EvidencePackage.build(
        claim="Peltier Stack reached 11.2% efficiency at 60C delta",
        concept=conflict.concept,
        contributor="alice-lab",
        kind="experiment",
        outcome="challenged",  # challenges the 80% interpretation
        measurements={"efficiency_pct": 11.2, "delta_t_c": 60},
        environment={"module": "TEC1-12706", "meter": "Fluke-87V"},
        reproduction_steps=(
            "mount module on 60C plate",
            "load with 3.3 ohm resistor",
            "log voltage/current for 10 minutes",
        ),
        artifacts=(Artifact(name="run.csv", uri="platform://alice-lab/run.csv"),),
    )
    replication = EvidencePackage.build(
        claim="independent re-run: 10.8% efficiency, same setup",
        concept=conflict.concept,
        contributor="bob-garage",
        kind="replication",
        outcome="challenged",
        replicates=original.id,
        measurements={"efficiency_pct": 10.8},
    )
    dissent = EvidencePackage.build(
        claim="my stack showed 79% (measurement setup unverified)",
        concept=conflict.concept,
        contributor="wonder-labs",
        kind="observation",
        outcome="supported",
    )
    resolved = board.resolve(proposal.id, [original, replication, dissent])
    print(f"  resolved: {resolved.resolution.outcome} "
          f"via {len(resolved.resolution.package_ids)} package(s)")

    print("\n=== 4. Nothing is hidden: the provenance tree ===")
    print(binder.why(conflict.concept))
    breakdown = ledger.confidence(conflict.concept)
    print(f"\n  contributors={breakdown.contributors}  "
          f"independent replications={breakdown.independent_replications}")
    print(f"  support={breakdown.support_weight}  "
          f"challenge={breakdown.challenge_weight}")

    print("\n=== 5. Belief history (every change has a reason) ===")
    for i, version in enumerate(graph.history(conflict.concept), start=1):
        print(f"  v{i}: confidence {version.confidence:.2f}")

    store.close()
    print(f"\nDone. Everything preserved under {workdir}")


if __name__ == "__main__":
    main()
