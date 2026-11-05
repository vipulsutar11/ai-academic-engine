import logging
from typing import List, Tuple, Optional
from app.models.exam import ExamPaper
from app.models.blueprint import ExamBlueprint
from app.models.curriculum import SubjectCurriculum
from app.validation.schema import SchemaValidator
from app.validation.marks import MarksValidator
from app.validation.difficulty import DifficultyValidator
from app.validation.topic import TopicValidator
from app.validation.duplicate import SemanticDuplicateDetector
from app.blueprint.validator import BlueprintValidator

logger = logging.getLogger(__name__)

class DeterministicPaperValidator:
    """Master deterministic validator running all structural, mark, difficulty, topic, and duplicate checks."""

    def __init__(self, duplicate_detector: Optional[SemanticDuplicateDetector] = None):
        self.duplicate_detector = duplicate_detector or SemanticDuplicateDetector()

    def validate_paper(
        self,
        paper: ExamPaper,
        blueprint: Optional[ExamBlueprint] = None,
        curriculum: Optional[SubjectCurriculum] = None
    ) -> Tuple[bool, List[str]]:
        all_errors: List[str] = []

        # 1. Structural Schema Validation for each question
        for sec in paper.sections:
            for q in sec.questions:
                q_passed, q_errs = SchemaValidator.validate_question_structure(q)
                if not q_passed:
                    all_errors.extend([f"Section '{sec.name}', Question '{q.id}': {e}" for e in q_errs])

        # 2. Marks Validation
        m_passed, m_errs = MarksValidator.validate_marks(paper)
        if not m_passed:
            all_errors.extend(m_errs)

        # 3. Blueprint Structural Validation (if blueprint provided)
        if blueprint:
            bp_passed, bp_errs = BlueprintValidator.validate_paper_against_blueprint(paper, blueprint)
            if not bp_passed:
                all_errors.extend(bp_errs)

            # 4. Difficulty Balance Validation
            d_passed, d_errs = DifficultyValidator.validate_difficulty_balance(
                paper, blueprint.difficulty_distribution
            )
            if not d_passed:
                all_errors.extend(d_errs)

        # 5. Topic Coverage & Alignment (if curriculum provided)
        if curriculum:
            t_passed, t_errs = TopicValidator.validate_topic_coverage(paper, curriculum)
            if not t_passed:
                all_errors.extend(t_errs)

        # 6. Intra-Paper Duplicate Detection (Vectorized Matrix Check)
        all_questions = [q for sec in paper.sections for q in sec.questions]
        dup_results = self.duplicate_detector.find_duplicates_in_paper(all_questions)
        for dup_q, orig_q, sim in dup_results:
            all_errors.append(
                f"Question '{dup_q.id}' stem is duplicate of '{orig_q.id}' (similarity score: {sim:.3f})."
            )

        passed = len(all_errors) == 0
        if not passed:
            logger.warning(f"Paper validation failed with {len(all_errors)} errors:\n" + "\n".join(all_errors))
        else:
            logger.info(f"Paper '{paper.title}' successfully passed all deterministic validations!")

        return passed, all_errors
