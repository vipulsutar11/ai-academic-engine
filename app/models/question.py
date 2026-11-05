from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.curriculum import DifficultyLevel, QuestionType

class MCQOption(BaseModel):
    key: str = Field(..., description="Option letter e.g. A, B, C, D")
    text: str = Field(..., description="Option content text")
    explanation: Optional[str] = Field(default=None, description="Explanation why option is correct or distractor")

class MarkingSchemeStep(BaseModel):
    step_number: int = Field(..., ge=1)
    description: str = Field(..., description="Description of the step or key point")
    marks: float = Field(..., gt=0, description="Marks assigned to this step")

class Question(BaseModel):
    id: str = Field(..., description="Unique Question ID")
    type: QuestionType
    stem: str = Field(..., description="The main text/statement of the question")
    options: Optional[List[MCQOption]] = Field(default=None, description="List of MCQ options (for MCQ/Assertion-Reason)")
    correct_answer: str = Field(..., description="Exact correct answer string or key")
    marking_scheme: List[MarkingSchemeStep] = Field(default_factory=list, description="Step-by-step marking breakdown")
    marks: float = Field(..., gt=0, description="Marks for this question")
    difficulty: DifficultyLevel
    chapter_id: str
    topic_id: str
    sub_topic: Optional[str] = None
    explanation: Optional[str] = Field(default=None, description="Detailed solution / explanation")
    blooms_level: str = Field(default="Application")

class AnswerKey(BaseModel):
    question_id: str
    question_type: QuestionType
    correct_answer: str
    marking_scheme: List[MarkingSchemeStep]
    explanation: Optional[str] = None

class QuestionBatch(BaseModel):
    questions: List[Question] = Field(default_factory=list, description="List of generated questions")

