from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from app.models.curriculum import BoardType, DifficultyLevel, QuestionType

class QuestionBlueprint(BaseModel):
    id: str = Field(..., description="Unique blueprint question ID, e.g. Q1_1")
    question_type: QuestionType
    marks: float = Field(..., gt=0)
    difficulty: DifficultyLevel
    chapter_id: str
    topic_id: str
    sub_topic: Optional[str] = None
    blooms_level: str = Field(default="Application", description="Knowledge, Understanding, Application, Analysis, Evaluation, Creation")

class SectionBlueprint(BaseModel):
    name: str = Field(..., description="e.g. Section A")
    instructions: str = Field(default="Answer all questions.")
    question_blueprints: List[QuestionBlueprint] = Field(..., min_length=1)
    total_marks: float = Field(..., gt=0)
    optional_questions_count: int = Field(default=0, ge=0)

class ExamBlueprint(BaseModel):
    title: str = Field(..., description="Exam title e.g. CBSE Class 10 Science Mid-Term Exam")
    board: BoardType
    academic_year: str = Field(..., description="e.g. 2024-25")
    class_name: str = Field(..., description="e.g. Class 10")
    subject: str = Field(..., description="e.g. Science")
    total_marks: float = Field(..., gt=0)
    duration_minutes: int = Field(..., gt=0)
    sections: List[SectionBlueprint] = Field(..., min_length=1)
    difficulty_distribution: Dict[DifficultyLevel, float] = Field(
        default_factory=lambda: {
            DifficultyLevel.EASY: 0.30,
            DifficultyLevel.MEDIUM: 0.50,
            DifficultyLevel.HARD: 0.20
        },
        description="Target ratio for easy, medium, hard questions"
    )
    topic_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of topic_id to target marks percentage or absolute marks"
    )
