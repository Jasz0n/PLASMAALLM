"""The contributor loop over HTTP, through the official client (M52).

Everything a frontend/integrator does, driven by ``allm.client`` — submit
knowledge, watch confidence move, read provenance, follow the live feed.
Runs offline against an in-process app so it is deterministic; against a
real deployment the only change is the constructor::

    from allm.client import AllmClient
    allm = AllmClient("https://api.example", token=os.environ["ALLM_API_TOKEN"])

    PYTHONPATH=src python3 examples/81_client_end_to_end.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    raise SystemExit("this example needs the api extras: pip install -e '.[api,dev]'")

from allm.api.app import create_app
from allm.api.security import StaticTokenVerifier
from allm.client import AllmClient, AllmError, Response

TOKEN = "demo-token-000000000000"


class InProcessTransport:
    """Route client calls to an in-process app (stands in for real HTTP)."""

    def __init__(self, app) -> None:
        self._tc = TestClient(app)

    def request(self, method, url, headers, body) -> Response:
        r = self._tc.request(method, url, headers=headers, content=body)
        return Response(r.status_code, r.content, dict(r.headers))


def main() -> None:
    app = create_app(
        Path(tempfile.mkdtemp(prefix="allm-client-")) / "api.sqlite3",
        verifier=StaticTokenVerifier(TOKEN),
    )
    allm = AllmClient("http://testserver", token=TOKEN, transport=InProcessTransport(app))

    print("=== M52: contributor loop through allm.client ===")
    print(f"  health={allm.health()['status']} ready={allm.ready()['status']} "
          f"wire={allm.wire()['wire_version']}")

    # 1. a contributor shares an explanation → the KDP distils concepts
    ingested = allm.submit_documents([
        {"name": "workshop-1",
         "text": "A plasma is an ionized gas. We call it the fourth state of matter. "
                 "Plasma conducts electricity because its electrons are free."},
    ])
    print(f"\n  submitted documents → {ingested['units']} knowledge unit(s); "
          f"graph now holds {[c['name'] for c in allm.concepts()]}")

    # 2. evidence moves belief — and only evidence does (creates the concept
    #    on first mention, source='evidence', confidence earned not asserted)
    result = allm.submit_evidence(
        claim="a plasma lamp lights when energized", concept="plasma",
        contributor="ada", outcome="supported",
    )
    breakdown = result["confidence"]
    print(f"  evidence {result['package_id'][:16]}… → 'plasma' confidence "
          f"{breakdown['value']:.2f} from {breakdown['contributors']} contributor(s), "
          f"{breakdown['independent_replications']} replication(s)")

    # 3. provenance is inspectable — nothing is hidden
    concept = allm.concept("plasma")
    print(f"\n  provenance for 'plasma':\n    {concept['provenance'].splitlines()[0]}")

    # 4. the live feed — what a frontend subscribes to
    print("\n  live feed:")
    for event in allm.catch_up():
        print(f"    #{event['seq']} {event['type']} {event['subject']}")

    # 5. errors arrive typed
    try:
        allm.concept("unknown-concept")
    except AllmError as err:
        print(f"\n  typed error on a bad read: {err.status} {err.type}")

    print("\nDone. Same code runs against a real deployment via AllmClient(url, token).")


if __name__ == "__main__":
    main()
