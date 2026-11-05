import logging
from typing import List
from app.models.question import Question, AnswerKey
from app.models.exam import ExamPaper

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generates complete Answer Keys and Step-by-Step Marking Schemes."""

    @classmethod
    def generate_answer_key_for_question(cls, question: Question) -> AnswerKey:
        return AnswerKey(
            question_id=question.id,
            question_type=question.type,
            correct_answer=question.correct_answer,
            marking_scheme=question.marking_scheme,
            explanation=question.explanation
        )

    @classmethod
    def generate_answer_keys_for_paper(cls, paper: ExamPaper) -> List[AnswerKey]:
        keys: List[AnswerKey] = []
        for sec in paper.sections:
            for q in sec.questions:
                keys.append(cls.generate_answer_key_for_question(q))
        return keys
