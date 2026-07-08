"""Extract MK teaching notes from cleaned Kids workshops via Ollama.

**Not for training** — digests are lossy summaries. Train on the full
cleaned transcripts from ``examples/13_kids_transcripts_kdp.py``::

    transcripts/Kids/cleaned/*.txt       (full workshop)
    transcripts/Kids/cleaned/mk/*.txt  (Mr Keshe only)

This example optionally builds a short curriculum index for human review.

    PYTHONPATH=src python3 examples/14_kids_teaching_digest.py
    ALLM_WORKSHOP=knowledgeSeekerWorkshop3.txt   # one file only
"""

from __future__ import annotations

import os
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.teaching_digest import digest_mk_teaching, render_digest
from allm.models import ModelSpec, load_model

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "transcripts" / "Kids" / "cleaned"
DIGEST_DIR = ROOT / "transcripts" / "Kids" / "digest"


def digest_writer():
    path = ROOT / "configs/models/ollama_digest_writer.yaml"
    spec = ModelSpec.from_yaml(path)
    device = os.environ.get("ALLM_DEVICE", spec.device)
    return load_model(spec.model_copy(update={"device": device}))


def targets() -> list[Path]:
    one = os.environ.get("ALLM_WORKSHOP")
    if one:
        return [CLEANED / one]
    return sorted(CLEANED.glob("*.txt"))


def main() -> None:
    setup_logging("INFO")
    if not CLEANED.is_dir():
        raise SystemExit(f"Run examples/13_kids_transcripts_kdp.py first — missing {CLEANED}")

    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    model = digest_writer()
    files = targets()
    print(f"\n=== MK teaching digest ({len(files)} workshop(s)) ===")
    print(f"  model: {model.spec.model_id}")

    for path in files:
        cleaned = path.read_text(encoding="utf-8")
        digest = digest_mk_teaching(path.name, cleaned, model)
        out = DIGEST_DIR / path.name.replace(".txt", ".md")
        out.write_text(render_digest(digest), encoding="utf-8")
        print(f"\n  {path.name}")
        print(f"    title: {digest.title}")
        print(f"    concepts: {len(digest.key_concepts)}  questions: {len(digest.follow_up_questions)}")
        print(f"    -> {out}")

    print(f"\nDigests written under {DIGEST_DIR}")


if __name__ == "__main__":
    main()
