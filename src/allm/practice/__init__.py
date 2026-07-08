"""Practice Engine: learning by doing (PRACTICE.md, Roadmap M48)."""

from allm.practice.evidence import (
    PRACTICE_CONTRIBUTOR,
    reproduction_conflict,
    run_claim,
    run_to_package,
)
from allm.practice.exam import (
    PRACTICE_KIND,
    description_samples,
    practice_samples,
    prediction_exam,
    prediction_question,
)
from allm.practice.contribution import (
    ApprovalError,
    Contribution,
    ContributionBoard,
)
from allm.practice.executor import SandboxExecutor, bind_variables
from allm.practice.feedback import (
    contribution_question,
    record_review_outcome,
    review_exam_result,
)
from allm.practice.repo_tasks import CandidatePatch, repo_test_procedure, trial_patch
from allm.practice.sweep import next_variable, record_sweep, run_sweep
from allm.practice.types import (
    PracticeProcedure,
    PracticeRun,
    SweepResult,
    VariableSpec,
)

__all__ = [
    "PRACTICE_CONTRIBUTOR",
    "PRACTICE_KIND",
    "ApprovalError",
    "CandidatePatch",
    "Contribution",
    "ContributionBoard",
    "PracticeProcedure",
    "PracticeRun",
    "SandboxExecutor",
    "SweepResult",
    "VariableSpec",
    "bind_variables",
    "contribution_question",
    "record_review_outcome",
    "repo_test_procedure",
    "review_exam_result",
    "trial_patch",
    "description_samples",
    "next_variable",
    "practice_samples",
    "prediction_exam",
    "prediction_question",
    "record_sweep",
    "reproduction_conflict",
    "run_claim",
    "run_sweep",
    "run_to_package",
]
