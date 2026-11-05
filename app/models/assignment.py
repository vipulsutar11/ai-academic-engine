from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.curriculum import BoardType
from app.models.question import Question, AnswerKey

class AssignmentRubricItem(BaseModel):
    criterion: str = Field(..., description="Rubric criterion e.g. Conceptual Understanding")
    description: str = Field(..., description="Detailed description of grading expectations")
    max_marks: float = Field(..., gt=0)

class Assignment(BaseModel):
    id: str = Field(..., description="Unique Assignment ID")
    title: str = Field(..., description="Assignment Title e.g. Electricity & Magnetism Practice Sheet")
    board: BoardType
    academic_year: str = Field(..., description="e.g. 2024-25")
    class_name: str = Field(..., description="e.g. Class 10")
    subject: str = Field(..., description="e.g. Science")
    total_marks: float = Field(..., gt=0)
    instructions: List[str] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    rubric: List[AssignmentRubricItem] = Field(default_factory=list)
    answer_key: List[AnswerKey] = Field(default_factory=list)
