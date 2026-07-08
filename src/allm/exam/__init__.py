"""Exams: shared question/answer vocabulary, generation and grading.

Phase 2 provides dataset-backed generation and deterministic grading.
Phase 7 (exam engine) extends this package with generative, coding and
cross-domain exams plus judge-based grading — behind the same
protocols.
"""

from allm.exam.base import (
    Answer,
    Exam,
    ExamGenerator,
    ExamResult,
    Question,
    QuestionResult,
    exam_generators,
)
from allm.exam.coding import CodingGrader, extract_code
from allm.exam.dataset_generator import DatasetExamGenerator
from allm.exam.generative import ModelExamGenerator, parse_questions
from allm.exam.paraphrase import paraphrase_definition_prompt
from allm.exam.paraphrase_generator import ParaphraseExamGenerator
from allm.exam.grading import CompositeGrader, EscalatingGrader, ExactMatchGrader, Grader, graders
from allm.exam.text import normalise
from allm.exam.llm_judge import LLMJudgeGrader, build_judge_prompt, parse_judge_response
from allm.exam.multi_judge import (
    MultiDimensionalGrader,
    build_multi_judge_prompt,
    multi_judge_enabled,
    parse_multi_judge_response,
)
from allm.exam.verdicts import MultiDimensionalVerdict
from allm.exam.disagreement_log import append_disagreements, collect_disagreements

__all__ = [
    "CodingGrader",
    "extract_code",
    "ModelExamGenerator",
    "parse_questions",
    "CompositeGrader",
    "EscalatingGrader",
    "Answer",
    "Exam",
    "ExamGenerator",
    "ExamResult",
    "Question",
    "QuestionResult",
    "exam_generators",
    "DatasetExamGenerator",
    "ParaphraseExamGenerator",
    "paraphrase_definition_prompt",
    "ExactMatchGrader",
    "LLMJudgeGrader",
    "MultiDimensionalGrader",
    "MultiDimensionalVerdict",
    "build_judge_prompt",
    "build_multi_judge_prompt",
    "multi_judge_enabled",
    "parse_judge_response",
    "parse_multi_judge_response",
    "append_disagreements",
    "collect_disagreements",
    "Grader",
    "graders",
    "normalise",
]
