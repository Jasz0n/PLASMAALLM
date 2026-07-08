"""Student identity: mission, domains, and specialization profile."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StudentIdentity(BaseModel):
    """Who a student is and what knowledge belongs in their mission."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    student_id: str = Field(alias="id")
    display_name: str = ""
    mission: str = ""
    core_domains: tuple[str, ...] = ()
    primary_domains: tuple[str, ...] = ()
    secondary_domains: tuple[str, ...] = ()
    ignored_domains: tuple[str, ...] = ()
    exploration_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    exploration_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    primary_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    secondary_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    core_weight: float = Field(default=1.0, ge=0.0, le=1.0)


def domain_matches(topic: str, domain: str) -> bool:
    """True when a topic name overlaps a domain label (hyphen/space tolerant)."""
    topic_norm = topic.lower().replace("-", " ").replace("_", " ")
    domain_norm = domain.lower().replace("-", " ").replace("_", " ")
    if domain_norm in topic_norm or topic_norm in domain_norm:
        return True
    return domain_norm.replace(" ", "") in topic_norm.replace(" ", "")


def exploration_roll(student_id: str, topic: str, seed: int = 0) -> float:
    """Deterministic pseudo-random roll in [0, 1) for exploration sampling."""
    payload = f"{seed}:{student_id}:{topic}".encode()
    digest = sum(payload) % 10_000
    return digest / 10_000


def domain_fit(
    topic: str,
    identity: StudentIdentity,
    *,
    seed: int = 0,
) -> tuple[float, str]:
    """Return an importance multiplier and a short reason for one topic."""
    roll = exploration_roll(identity.student_id, topic, seed)

    if any(domain_matches(topic, domain) for domain in identity.ignored_domains):
        if roll < identity.exploration_rate:
            return identity.exploration_weight, "exploration on ignored domain"
        return 0.0, "ignored domain"

    if any(domain_matches(topic, domain) for domain in identity.core_domains):
        return identity.core_weight, "shared core"

    if any(domain_matches(topic, domain) for domain in identity.primary_domains):
        return identity.primary_weight, "primary mission"

    if any(domain_matches(topic, domain) for domain in identity.secondary_domains):
        return identity.secondary_weight, "secondary domain"

    if roll < identity.exploration_rate:
        return identity.exploration_weight, "exploration"

    return 0.0, "outside mission"


def load_shared_core(path: Path | str) -> tuple[str, ...]:
    """Load shared core domain list from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    domains = data.get("domains", ())
    if not isinstance(domains, list):
        raise ValueError(f"shared core {path} must list domains")
    return tuple(str(domain) for domain in domains)


def load_identity(path: Path | str) -> StudentIdentity:
    """Load one student identity; resolves ``core_from`` relative to the file."""
    file_path = Path(path)
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"identity {path} must be a mapping")

    core_from = data.pop("core_from", None)
    identity = StudentIdentity.model_validate(data)
    if core_from is None:
        return identity

    core_path = file_path.parent / str(core_from)
    merged_core = tuple(dict.fromkeys((*load_shared_core(core_path), *identity.core_domains)))
    return identity.model_copy(update={"core_domains": merged_core})


def load_identities_dir(path: Path | str) -> dict[str, StudentIdentity]:
    """Load every ``*_student.yaml`` identity in a directory."""
    directory = Path(path)
    identities: dict[str, StudentIdentity] = {}
    for file_path in sorted(directory.glob("*_student.yaml")):
        identity = load_identity(file_path)
        identities[identity.student_id] = identity
    return identities
