from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.models.curriculum import BoardType, DifficultyLevel, QuestionType
from app.models.blueprint import ExamBlueprint, QuestionBlueprint

class ExamGenerationRequest(BaseModel):
    prompt: Optional[str] = Field(default="10 mcq for chapter 1 of history", description="Prompt specifying question types, count, and subject e.g. '10 mcq for chapter 1 of history' or '5 short questions for science'")
    class_name: Optional[str] = Field(default="Class 10", description="e.g. Class 10 (auto-inferred from prompt if omitted)")
    subject: Optional[str] = Field(default=None, description="e.g. History, Science (auto-inferred from prompt if omitted)")
    board: Optional[BoardType] = Field(default=BoardType.CBSE, description="Board (CBSE, ICSE, UP_BOARD)")
    academic_year: str = Field(default="2024-25", description="Academic year e.g. 2024-25")
    language_mode: Optional[str] = Field(default="ENGLISH", description="Language mode: ENGLISH, HINDI, or BILINGUAL")
    total_marks: Optional[float] = Field(default=80.0, gt=0)
    duration_minutes: Optional[int] = Field(default=180, gt=0)
    title: Optional[str] = None
    custom_blueprint: Optional[ExamBlueprint] = None
    chapter_ids: Optional[List[str]] = None
    topic_ids: Optional[List[str]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "10 mcq for chapter 1 of history"
            }
        }
    }

class BatchExamGenerationRequest(BaseModel):
    requests: List[ExamGenerationRequest] = Field(..., description="List of exam paper requests e.g. Class 1 to Class 10 across subjects")
    max_workers: int = Field(default=10, ge=1, le=20, description="Parallel worker count for concurrent execution")

class QuestionGenerationRequest(BaseModel):
    board: BoardType
    academic_year: str = "2024-25"
    class_name: str
    subject: str
    chapter_id: str
    topic_id: str
    question_type: QuestionType = QuestionType.SHORT_ANSWER
    marks: float = 3.0
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    language_mode: str = Field(default="ENGLISH", description="Language mode: ENGLISH, HINDI, or BILINGUAL")
    prompt: Optional[str] = Field(default=None, description="Custom prompt or instructions for question generation")

class QuestionRegenerationRequest(BaseModel):
    board: BoardType
    academic_year: str = "2024-25"
    class_name: str
    subject: str
    blueprint: QuestionBlueprint
    rejection_reason: str = Field(..., description="Reason why original question failed validation or quality check")
    prompt: Optional[str] = Field(default=None, description="Custom user prompt instructions")

class AssignmentGenerationRequest(BaseModel):
    board: BoardType
    academic_year: str = "2024-25"
    class_name: str
    subject: str
    total_marks: float = 20.0
    title: Optional[str] = None
    chapter_ids: Optional[List[str]] = None
    prompt: Optional[str] = Field(default=None, description="Custom prompt or instructions for assignment generation")

class CurriculumIngestRequest(BaseModel):
    board: BoardType
    academic_year: str
    class_name: str
    subject: str
    content: str = Field(..., description="Curriculum raw text or json content to index")
    document_name: str = Field(default="curriculum_doc")

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
