"""Database maintenance: verify and restore (Roadmap M50).

Backups are taken online through :meth:`SQLiteRecordStore.backup_to`
(SQLite's backup API — consistent even mid-write). Restore is a
file-level operation on a *closed* database: verify the backup, refuse
to clobber an existing target unless forced, and keep the displaced
file as ``<name>.replaced`` — nothing is ever silently destroyed.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from allm.core.logging import get_logger

logger = get_logger("storage.maintenance")


def verify_database(path: Path | str) -> tuple[bool, str]:
    """Run SQLite's integrity check; returns (ok, detail)."""
    db = Path(path)
    if not db.is_file():
        return False, f"no database at {db}"
    conn = sqlite3.connect(db)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            return False, result
        count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        return True, f"ok ({count} records)"
    except sqlite3.DatabaseError as exc:
        return False, str(exc)
    finally:
        conn.close()


def restore_database(
    backup_path: Path | str, target_path: Path | str, *, force: bool = False
) -> Path:
    """Put a verified backup in place of ``target_path``.

    Refuses when the target exists and ``force`` is not given; with
    force, the displaced database survives as ``<target>.replaced``.
    """
    backup = Path(backup_path)
    target = Path(target_path)
    ok, detail = verify_database(backup)
    if not ok:
        raise ValueError(f"backup failed verification: {detail}")
    if target.exists():
        if not force:
            raise FileExistsError(
                f"{target} exists — pass force to replace it (the old file "
                "is kept as .replaced)"
            )
        displaced = target.with_suffix(target.suffix + ".replaced")
        shutil.move(target, displaced)
        logger.info("displaced %s -> %s", target, displaced)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, target)
    logger.info("restored %s from %s (%s)", target, backup, detail)
    return target
