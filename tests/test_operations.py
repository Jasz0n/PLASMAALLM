"""Operational surface: audit trail + backup/restore (M50), offline."""

from pathlib import Path

import pytest

from allm.cli.main import main
from allm.storage import SQLiteRecordStore
from allm.storage.maintenance import restore_database, verify_database


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "ops.sqlite3"
    store = SQLiteRecordStore(path)
    store.put("concepts", "plasma", {"v": 1}, reason="initial")
    store.put("concepts", "plasma", {"v": 2}, reason="revised: better evidence")
    store.put("evidence_packages", "pkg_1", {"claim": "x"}, reason="experiment by ada")
    store.close()
    return path


def test_audit_reads_every_write_newest_first(db: Path) -> None:
    store = SQLiteRecordStore(db)
    try:
        trail = store.audit()
        assert [(r.namespace, r.key, r.version) for r in trail] == [
            ("evidence_packages", "pkg_1", 1),
            ("concepts", "plasma", 2),
            ("concepts", "plasma", 1),
        ]
        assert trail[1].reason == "revised: better evidence"
        only_concepts = store.audit("concepts")
        assert {r.namespace for r in only_concepts} == {"concepts"}
        assert len(store.audit(limit=1)) == 1
        assert store.audit(limit=1, offset=1)[0].key == "plasma"
    finally:
        store.close()


def test_backup_verify_restore_roundtrip(tmp_path: Path, db: Path) -> None:
    backup = tmp_path / "backups" / "ops.backup.sqlite3"
    store = SQLiteRecordStore(db)
    try:
        store.backup_to(backup)
        # the backup is consistent even while the source keeps writing
        store.put("concepts", "plasma", {"v": 3}, reason="post-backup write")
    finally:
        store.close()

    ok, detail = verify_database(backup)
    assert ok and "3 records" in detail

    # restore refuses to clobber silently
    with pytest.raises(FileExistsError, match="force"):
        restore_database(backup, db)

    restored = restore_database(backup, db, force=True)
    assert restored == db
    assert db.with_suffix(".sqlite3.replaced").exists()  # nothing destroyed
    store = SQLiteRecordStore(db)
    try:
        assert store.get("concepts", "plasma").value == {"v": 2}  # pre-backup state
    finally:
        store.close()


def test_restore_rejects_corrupt_backups(tmp_path: Path) -> None:
    fake = tmp_path / "corrupt.sqlite3"
    fake.write_text("this is not a database")
    ok, detail = verify_database(fake)
    assert not ok
    with pytest.raises(ValueError, match="verification"):
        restore_database(fake, tmp_path / "target.sqlite3")


def test_cli_audit_and_db_commands(tmp_path: Path, db: Path, capsys) -> None:
    assert main(["audit", "--db", str(db), "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "concepts/plasma" in out and "experiment by ada" in out

    backup = tmp_path / "cli.backup.sqlite3"
    assert main(["db", "backup", "--db", str(db), str(backup)]) == 0
    assert "3 records" in capsys.readouterr().out
    assert main(["db", "verify", "--db", str(backup)]) == 0
    # refusal without --force is an error exit, not silence
    assert main(["db", "restore", "--db", str(db), str(backup)]) == 1
    assert main(["db", "restore", "--db", str(db), str(backup), "--force"]) == 0


def test_api_audit_endpoint(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import AllowAllVerifier

    app = create_app(tmp_path / "api.sqlite3", verifier=AllowAllVerifier())
    with TestClient(app) as client:
        client.post(
            "/evidence",
            json={
                "claim": "c",
                "concept": "X",
                "contributor": "ada",
                "outcome": "supported",
            },
        )
        trail = client.get("/audit", params={"limit": 5}).json()
        assert trail and {"namespace", "key", "version", "reason", "created_at"} <= set(trail[0])
        assert all("value" not in row for row in trail)
