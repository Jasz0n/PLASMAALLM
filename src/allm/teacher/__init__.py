"""Teacher: evaluates students, assigns learning goals, creates exams,
measures progress, maintains global knowledge state.

Public API::

    teacher = Teacher(KnowledgeState(store), exam_generator, grader)
    exam    = teacher.create_exam(topics=["math"], num_questions=5)
    result  = teacher.evaluate(student, exam)
    goals   = teacher.assign_goals(student.student_id)
    report  = teacher.progress(student.student_id)
"""

from allm.teacher.consultation import ConsultationRequest, consultation_samples, request_consultation
from allm.teacher.mediated_consultation import MediatedConsultationResult, mediated_consultation
from allm.teacher.experts import ExpertRanking, best_expert, rank_experts
from allm.teacher.state import KnowledgeState
from allm.teacher.teacher import Teacher
from allm.teacher.types import LearningGoal, ProgressReport, TeacherConfig, TopicProgress

__all__ = [
    "KnowledgeState",
    "Teacher",
    "ExpertRanking",
    "ConsultationRequest",
    "consultation_samples",
    "MediatedConsultationResult",
    "mediated_consultation",
    "ShowMeResult",
    "teacher_show_me",
    "ConsultationEvidence",
    "consultation_show_me",
    "derive_show_me_query",
    "request_consultation",
    "best_expert",
    "rank_experts",
    "LearningGoal",
    "ProgressReport",
    "TeacherConfig",
    "TopicProgress",
]


def __getattr__(name: str):
    if name == "ShowMeResult":
        from allm.teacher.show_me import ShowMeResult

        return ShowMeResult
    if name == "teacher_show_me":
        from allm.teacher.show_me import teacher_show_me

        return teacher_show_me
    if name == "ConsultationEvidence":
        from allm.teacher.show_me import ConsultationEvidence

        return ConsultationEvidence
    if name == "consultation_show_me":
        from allm.teacher.show_me import consultation_show_me

        return consultation_show_me
    if name == "derive_show_me_query":
        from allm.teacher.show_me import derive_show_me_query

        return derive_show_me_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
