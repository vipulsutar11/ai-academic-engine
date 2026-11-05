import logging
import uuid
from typing import List, Optional
from app.models.assignment import Assignment, AssignmentRubricItem
from app.models.curriculum import BoardType, QuestionType, DifficultyLevel
from app.models.blueprint import QuestionBlueprint
from app.ai.provider import BaseLLMProvider
from app.generators.question import QuestionGenerator
from app.generators.answer import AnswerGenerator
from app.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

class AssignmentGenerator:
    """Generates structured practice/homework assignments with grading rubrics."""

    def __init__(self, provider: BaseLLMProvider, retriever: Optional[RAGRetriever] = None):
        self.question_generator = QuestionGenerator(provider=provider, retriever=retriever)

    def generate_assignment(
        self,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        total_marks: float = 20.0,
        title: Optional[str] = None,
        chapter_ids: Optional[List[str]] = None
    ) -> Assignment:
        assignment_id = f"ASSIGN_{uuid.uuid4().hex[:8]}"
        assign_title = title or f"{board.value} {class_name} {subject} Homework Assignment"

        ch_ids = chapter_ids or ["CH_01"]
        
        # Build question blueprints for assignment
        q_blueprints = [
            QuestionBlueprint(
                id=f"Q_ASSIGN_1",
                question_type=QuestionType.SHORT_ANSWER,
                marks=5.0,
                difficulty=DifficultyLevel.EASY,
                chapter_id=ch_ids[0],
                topic_id=f"{ch_ids[0]}_TOPIC_01"
            ),
            QuestionBlueprint(
                id=f"Q_ASSIGN_2",
                question_type=QuestionType.SHORT_ANSWER,
                marks=5.0,
                difficulty=DifficultyLevel.MEDIUM,
                chapter_id=ch_ids[0],
                topic_id=f"{ch_ids[0]}_TOPIC_01"
            ),
            QuestionBlueprint(
                id=f"Q_ASSIGN_3",
                question_type=QuestionType.LONG_ANSWER,
                marks=10.0,
                difficulty=DifficultyLevel.HARD,
                chapter_id=ch_ids[0],
                topic_id=f"{ch_ids[0]}_TOPIC_01"
            )
        ]

        questions = []
        for bp in q_blueprints:
            q = self.question_generator.generate_question(
                bp=bp,
                board=board,
                academic_year=academic_year,
                class_name=class_name,
                subject=subject
            )
            questions.append(q)

        answer_keys = [AnswerGenerator.generate_answer_key_for_question(q) for q in questions]

        rubric = [
            AssignmentRubricItem(
                criterion="Conceptual Accuracy",
                description="Correct application of underlying physical principles and formulas.",
                max_marks=total_marks * 0.6
            ),
            AssignmentRubricItem(
                criterion="Step-by-Step Working & Neatness",
                description="Logical sequence of working, correct units, and clear diagrams.",
                max_marks=total_marks * 0.4
            )
        ]

        logger.info(f"Generated assignment '{assign_title}' with {len(questions)} questions.")

        return Assignment(
            id=assignment_id,
            title=assign_title,
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            total_marks=total_marks,
            instructions=[
                "Answer all questions in full detail.",
                "Show clear mathematical derivations and steps where applicable."
            ],
            questions=questions,
            rubric=rubric,
            answer_key=answer_keys
        )
