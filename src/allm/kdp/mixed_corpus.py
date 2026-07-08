"""Mixed plasma + software corpus loading for dual-specialist experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.data.base import Sample
from allm.kdp.corpus import load_samples_jsonl
from allm.kdp.curriculum import load_curriculum_splits
from allm.students.identity import StudentIdentity, domain_fit


DEFAULT_SOFTWARE_PATH = Path("transcripts/Software/samples_dev.jsonl")


class DomainCorpus(BaseModel):
    """Train and holdout pools for one domain."""

    model_config = ConfigDict(frozen=True)

    domain: str
    train: tuple[Sample, ...]
    holdout: tuple[Sample, ...]


class MixedCorpus(BaseModel):
    """Plasma workshop + software dev pools for specialist experiments."""

    model_config = ConfigDict(frozen=True)

    plasma: DomainCorpus
    software: DomainCorpus

    @property
    def merged_train(self) -> list[Sample]:
        return list(self.plasma.train) + list(self.software.train)

    @property
    def merged_holdout(self) -> list[Sample]:
        return list(self.plasma.holdout) + list(self.software.holdout)


def split_samples(
    samples: list[Sample],
    *,
    holdout_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic train/holdout split for a sample list."""
    if not samples:
        return [], []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    holdout_count = max(1, int(len(shuffled) * holdout_fraction))
    holdout = shuffled[:holdout_count]
    train = shuffled[holdout_count:]
    if not train:
        train, holdout = holdout[:-1], holdout[-1:]
    return train, holdout


def load_software_samples(root: Path) -> list[Sample]:
    """Load labelled software fixture samples."""
    override = os.environ.get("ALLM_SOFTWARE_SAMPLES", "").strip()
    path = Path(override) if override else root / DEFAULT_SOFTWARE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"software corpus missing: {path}")
    return [sample for sample in load_samples_jsonl(path) if sample.target]


def load_mixed_corpus(root: Path, *, software_holdout_fraction: float = 0.25) -> MixedCorpus:
    """Load plasma curriculum splits and software fixture with holdout."""
    plasma_train, plasma_holdout = load_curriculum_splits(root)
    software_all = load_software_samples(root)
    software_train, software_holdout = split_samples(
        software_all,
        holdout_fraction=software_holdout_fraction,
        seed=int(os.environ.get("ALLM_LOOP_SEED", "42")),
    )
    return MixedCorpus(
        plasma=DomainCorpus(
            domain="plasma",
            train=tuple(plasma_train),
            holdout=tuple(plasma_holdout),
        ),
        software=DomainCorpus(
            domain="software",
            train=tuple(software_train),
            holdout=tuple(software_holdout),
        ),
    )


def samples_for_identity(
    samples: list[Sample],
    identity: StudentIdentity,
    *,
    seed: int = 0,
) -> list[Sample]:
    """Filter samples to those matching a specialist mission."""
    kept: list[Sample] = []
    for sample in samples:
        topic = str(sample.metadata.get("topic", "general"))
        fit, _reason = domain_fit(topic, identity, seed=seed)
        if fit > 0.0:
            kept.append(sample)
    return kept


def contamination_rate(
    studied_ids: set[str],
    samples: list[Sample],
    identity: StudentIdentity,
    *,
    seed: int = 0,
) -> float:
    """Fraction of studied samples outside the student's mission."""
    if not studied_ids:
        return 0.0
    by_id = {sample.id: sample for sample in samples}
    outside = 0
    for sample_id in studied_ids:
        sample = by_id.get(sample_id)
        if sample is None:
            continue
        topic = str(sample.metadata.get("topic", "general"))
        fit, _ = domain_fit(topic, identity, seed=seed)
        if fit <= 0.0:
            outside += 1
    return outside / len(studied_ids)
