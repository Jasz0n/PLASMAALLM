"""Students: the learner side of the teacher/student architecture.

- :class:`Student` protocol — all the teacher ever sees.
- :class:`ScriptedStudent` — deterministic test double (Phase 2).
- :class:`ModelStudent` — wraps any LanguageModel, studies notes,
  self-reports confidence (Phase 3).
- :class:`FailureLog` — versioned failure storage, convertible back
  into training samples (Phase 3).
"""

from allm.students.base import Student, student_types
from allm.students.confidence import CONFIDENCE_INSTRUCTION, parse_confidence
from allm.students.failures import FailureLog, FailureRecord
from allm.students.identity import (
    StudentIdentity,
    domain_fit,
    domain_matches,
    load_identities_dir,
    load_identity,
    load_shared_core,
)
from allm.students.model_student import ModelStudent, ModelStudentConfig
from allm.students.scripted import ScriptedStudent

__all__ = [
    "Student",
    "student_types",
    "ScriptedStudent",
    "ModelStudent",
    "ModelStudentConfig",
    "FailureLog",
    "FailureRecord",
    "StudentIdentity",
    "domain_fit",
    "domain_matches",
    "load_identities_dir",
    "load_identity",
    "load_shared_core",
    "CONFIDENCE_INSTRUCTION",
    "parse_confidence",
]
