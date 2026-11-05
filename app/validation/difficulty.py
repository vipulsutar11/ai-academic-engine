import logging
from typing import List, Dict, Tuple
from app.models.curriculum import DifficultyLevel
from app.models.blueprint import ExamBlueprint
from app.models.exam import ExamPaper

logger = logging.getLogger(__name__)

class DifficultyValidator:
    """Validates difficulty level distribution against target ratios."""

    @classmethod
    def validate_difficulty_balance(
        cls,
        paper: ExamPaper,
        target_distribution: Dict[DifficultyLevel, float],
        tolerance: float = 0.15
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        diff_marks: Dict[DifficultyLevel, float] = {
            DifficultyLevel.EASY: 0.0,
            DifficultyLevel.MEDIUM: 0.0,
            DifficultyLevel.HARD: 0.0
        }

        total_marks = paper.total_marks or 1.0

        for sec in paper.sections:
            for q in sec.questions:
                diff_marks[q.difficulty] = diff_marks.get(q.difficulty, 0.0) + q.marks

        actual_ratios = {d: round(m / total_marks, 3) for d, m in diff_marks.items()}

        for diff_level, target_pct in target_distribution.items():
            actual_pct = actual_ratios.get(diff_level, 0.0)
            if abs(actual_pct - target_pct) > tolerance:
                errors.append(
                    f"Difficulty '{diff_level.value}' ratio is {actual_pct:.2f}, which deviates from target {target_pct:.2f} (tolerance +/-{tolerance})."
                )

        passed = len(errors) == 0
        return passed, errors
