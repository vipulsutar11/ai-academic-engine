import logging
from typing import List, Dict, Tuple
from app.models.blueprint import ExamBlueprint
from app.models.exam import ExamPaper

logger = logging.getLogger(__name__)

class BlueprintValidator:
    """Validates that a generated ExamPaper strictly satisfies an ExamBlueprint."""

    @classmethod
    def validate_paper_against_blueprint(
        cls,
        paper: ExamPaper,
        blueprint: ExamBlueprint
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        # 1. Total marks match
        if abs(paper.total_marks - blueprint.total_marks) > 1e-3:
            errors.append(
                f"Paper total marks ({paper.total_marks}) does not match blueprint target total ({blueprint.total_marks})."
            )

        # 2. Section count match
        if len(paper.sections) != len(blueprint.sections):
            errors.append(
                f"Paper section count ({len(paper.sections)}) does not match blueprint section count ({len(blueprint.sections)})."
            )

        # 3. Individual section question count & section mark match
        for bp_sec in blueprint.sections:
            matching_paper_sec = next((s for s in paper.sections if s.name == bp_sec.name), None)
            if not matching_paper_sec:
                errors.append(f"Missing section '{bp_sec.name}' in generated paper.")
                continue

            expected_q_count = len(bp_sec.question_blueprints)
            actual_q_count = len(matching_paper_sec.questions)
            if actual_q_count != expected_q_count:
                errors.append(
                    f"Section '{bp_sec.name}' question count ({actual_q_count}) does not match blueprint requirement ({expected_q_count})."
                )

            if abs(matching_paper_sec.total_marks - bp_sec.total_marks) > 1e-3:
                errors.append(
                    f"Section '{bp_sec.name}' total marks ({matching_paper_sec.total_marks}) does not match blueprint section target ({bp_sec.total_marks})."
                )

            # Check individual questions marks & types
            for i, bp_q in enumerate(bp_sec.question_blueprints):
                if i < len(matching_paper_sec.questions):
                    p_q = matching_paper_sec.questions[i]
                    if abs(p_q.marks - bp_q.marks) > 1e-3:
                        errors.append(
                            f"Question '{p_q.id}' in '{bp_sec.name}' has marks {p_q.marks}, expected {bp_q.marks}."
                        )
                    if p_q.type != bp_q.question_type:
                        errors.append(
                            f"Question '{p_q.id}' in '{bp_sec.name}' type {p_q.type} does not match blueprint type {bp_q.question_type}."
                        )

        passed = len(errors) == 0
        if not passed:
            logger.warning(f"Paper-Blueprint validation failed with {len(errors)} errors.")
        else:
            logger.info("Paper successfully validated against blueprint.")

        return passed, errors
