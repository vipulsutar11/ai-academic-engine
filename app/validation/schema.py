import logging
from typing import List, Tuple
from app.models.question import Question
from app.models.curriculum import QuestionType

logger = logging.getLogger(__name__)

class SchemaValidator:
    """Validates structural Pydantic integrity and question field constraints."""

    @classmethod
    def validate_question_structure(cls, question: Question) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        # Stem non-empty
        if not question.stem or len(question.stem.strip()) < 5:
            errors.append(f"Question '{question.id}' stem is empty or too short.")

        # MCQ validation
        if question.type == QuestionType.MCQ:
            if not question.options or len(question.options) != 4:
                errors.append(f"MCQ '{question.id}' must have exactly 4 options (A, B, C, D). Found: {len(question.options) if question.options else 0}.")
            else:
                option_keys = [opt.key.upper() for opt in question.options]
                if sorted(option_keys) != ["A", "B", "C", "D"]:
                    errors.append(f"MCQ '{question.id}' option keys must be ['A', 'B', 'C', 'D']. Found: {option_keys}.")
                
                # Verify correct_answer points to a valid option key
                if question.correct_answer.upper() not in ["A", "B", "C", "D"]:
                    errors.append(f"MCQ '{question.id}' correct_answer ('{question.correct_answer}') must be one of ['A', 'B', 'C', 'D'].")

                # Verify non-empty text for options
                for opt in question.options:
                    if not opt.text or not opt.text.strip():
                        errors.append(f"MCQ '{question.id}' option '{opt.key}' text is empty.")

        # Marking scheme sum check
        if question.marking_scheme:
            step_marks_sum = sum(step.marks for step in question.marking_scheme)
            if abs(step_marks_sum - question.marks) > 1e-3:
                errors.append(
                    f"Question '{question.id}' marking scheme sum ({step_marks_sum}) does not equal question marks ({question.marks})."
                )

        passed = len(errors) == 0
        return passed, errors
