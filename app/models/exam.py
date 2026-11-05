from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from app.models.curriculum import BoardType
from app.models.question import Question, AnswerKey

class ExamSection(BaseModel):
    name: str = Field(..., description="Section Name e.g. Section A")
    instructions: str = Field(default="Answer all questions in this section.")
    questions: List[Question] = Field(default_factory=list)
    total_marks: float = Field(..., ge=0)

class ExamPaper(BaseModel):
    id: str = Field(..., description="Unique Exam Paper ID")
    title: str = Field(..., description="Paper Title e.g. Class 10 Physics Annual Exam 2024-25")
    board: BoardType
    academic_year: str = Field(..., description="Academic Year e.g. 2024-25")
    class_name: str = Field(..., description="e.g. Class 10")
    subject: str = Field(..., description="e.g. Science")
    total_marks: float = Field(..., gt=0)
    duration_minutes: int = Field(..., gt=0)
    general_instructions: List[str] = Field(default_factory=list)
    sections: List[ExamSection] = Field(default_factory=list)
    answer_key: List[AnswerKey] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
