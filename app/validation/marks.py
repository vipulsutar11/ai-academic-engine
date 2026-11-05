import logging
from typing import List, Tuple
from app.models.exam import ExamPaper

logger = logging.getLogger(__name__)

class MarksValidator:
    """Validates deterministic mark sums for questions, sections, and full exam paper."""

    @classmethod
    def validate_marks(cls, paper: ExamPaper) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        calculated_paper_total = 0.0

        for sec in paper.sections:
            calculated_sec_total = 0.0
            for q in sec.questions:
                if q.marks <= 0:
                    errors.append(f"Question '{q.id}' in '{sec.name}' has invalid non-positive marks ({q.marks}).")
                calculated_sec_total += q.marks

            if abs(calculated_sec_total - sec.total_marks) > 1e-3:
                errors.append(
                    f"Section '{sec.name}' calculated marks sum ({calculated_sec_total}) does not match declared total_marks ({sec.total_marks})."
                )

            calculated_paper_total += calculated_sec_total

        if abs(calculated_paper_total - paper.total_marks) > 1e-3:
            errors.append(
                f"Paper calculated total marks ({calculated_paper_total}) does not match declared total_marks ({paper.total_marks})."
            )

        passed = len(errors) == 0
        return passed, errors
