import logging
from typing import Optional, List
from app.models.question import Question
from app.models.blueprint import QuestionBlueprint
from app.models.curriculum import BoardType, QuestionType, DifficultyLevel
from app.ai.provider import BaseLLMProvider
from app.generators.question import QuestionGenerator
from app.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

class MCQGenerator:
    """Specialized MCQ generator ensuring valid distractor sets and correct option format."""

    def __init__(self, provider: BaseLLMProvider, retriever: Optional[RAGRetriever] = None):
        self.question_generator = QuestionGenerator(provider=provider, retriever=retriever)

    def generate_mcq(
        self,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        chapter_id: str,
        topic_id: str,
        marks: float = 1.0,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        existing_questions: Optional[List[str]] = None
    ) -> Question:
        bp = QuestionBlueprint(
            id=f"MCQ_{chapter_id}_{topic_id}",
            question_type=QuestionType.MCQ,
            marks=marks,
            difficulty=difficulty,
            chapter_id=chapter_id,
            topic_id=topic_id,
            blooms_level="Knowledge" if difficulty == DifficultyLevel.EASY else "Application"
        )

        return self.question_generator.generate_question(
            bp=bp,
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            existing_questions_summary=existing_questions
        )
