import logging
from typing import List, Tuple, Optional
from app.models.exam import ExamPaper
from app.models.curriculum import SubjectCurriculum

logger = logging.getLogger(__name__)

class TopicValidator:
    """Validates topic coverage and alignment with SubjectCurriculum."""

    @classmethod
    def validate_topic_coverage(
        cls,
        paper: ExamPaper,
        curriculum: SubjectCurriculum
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        valid_chapter_ids = {ch.id for ch in curriculum.chapters}
        valid_topic_ids = {t.id for ch in curriculum.chapters for t in ch.topics}

        used_topic_ids = set()

        for sec in paper.sections:
            for q in sec.questions:
                if q.chapter_id not in valid_chapter_ids:
                    errors.append(f"Question '{q.id}' references unknown chapter_id '{q.chapter_id}'.")
                if q.topic_id not in valid_topic_ids:
                    errors.append(f"Question '{q.id}' references unknown topic_id '{q.topic_id}'.")
                else:
                    used_topic_ids.add(q.topic_id)

        # Check coverage percentage
        coverage_pct = len(used_topic_ids) / len(valid_topic_ids) if valid_topic_ids else 1.0
        logger.info(f"Topic coverage for paper '{paper.title}': {coverage_pct * 100:.1f}% ({len(used_topic_ids)}/{len(valid_topic_ids)} topics covered).")

        passed = len(errors) == 0
        return passed, errors
