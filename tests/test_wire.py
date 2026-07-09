"""The frozen wire contract (M51): generated, versioned, drift-guarded."""

import json
from pathlib import Path

import pytest

from allm.wire import WIRE_VERSION, wire_contract


def test_contract_covers_the_public_surface() -> None:
    contract = wire_contract()
    assert contract["wire_version"] == WIRE_VERSION
    # the request a contributor actually sends, with its required fields
    ev = contract["requests"]["EvidenceSubmission"]
    assert set(ev["required"]) == {"claim", "concept", "contributor", "outcome"}
    # vocabularies are listed so a client can map, not hard-reject
    assert "supported" in contract["vocabularies"]["Outcome"]
    assert "replication" in contract["vocabularies"]["PackageKind"]
    # the response shapes and the event feed entry are all present
    assert {"ConfidenceBreakdown", "ConceptSummary", "Event"} <= set(contract["responses"])
    assert "seq" in contract["responses"]["Event"]["properties"]


def test_contract_is_json_serialisable() -> None:
    # platform teams consume this as JSON; it must round-trip cleanly
    assert json.loads(json.dumps(wire_contract()))["wire_version"] == WIRE_VERSION


def test_published_wire_contract_is_current() -> None:
    """docs/wire-format.json is the frozen contract — regenerating must
    reproduce the same version and schema surface, or it is stale."""
    published = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "wire-format.json").read_text()
    )
    live = wire_contract()
    assert published["wire_version"] == live["wire_version"]
    assert published["requests"] == live["requests"]
    assert published["responses"] == live["responses"]
    assert published["vocabularies"] == live["vocabularies"]


def test_wire_endpoint_serves_the_contract(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app

    # open read: platform can discover the contract before authenticating
    app = create_app(tmp_path / "api.sqlite3")
    with TestClient(app) as client:
        body = client.get("/wire").json()
        assert body["wire_version"] == WIRE_VERSION
        assert "EvidenceSubmission" in body["requests"]


def test_cli_wire_export(tmp_path: Path, capsys) -> None:
    from allm.cli.main import main

    assert main(["wire"]) == 0
    assert '"wire_version"' in capsys.readouterr().out

    out = tmp_path / "contract.json"
    assert main(["wire", "--output", str(out)]) == 0
    assert json.loads(out.read_text())["wire_version"] == WIRE_VERSION
