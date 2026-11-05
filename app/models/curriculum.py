from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# Type alias for AcademicYear
AcademicYear = str

class BoardType(str, Enum):
    CBSE = "CBSE"
    ICSE = "ICSE"
    UP_BOARD = "UP_BOARD"

class DifficultyLevel(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"

class QuestionType(str, Enum):
    MCQ = "MCQ"
    SHORT_ANSWER = "SHORT_ANSWER"
    LONG_ANSWER = "LONG_ANSWER"
    NUMERICAL = "NUMERICAL"
    ASSERTION_REASON = "ASSERTION_REASON"
    CASE_STUDY = "CASE_STUDY"

class Topic(BaseModel):
    id: str = Field(..., description="Unique identifier for the topic")
    name: str = Field(..., description="Topic name")
    weightage: float = Field(default=1.0, ge=0.0, description="Relative weightage of topic in curriculum")
    keywords: List[str] = Field(default_factory=list, description="Keywords for RAG retrieval")
    learning_outcomes: List[str] = Field(default_factory=list, description="Expected learning outcomes")

class Chapter(BaseModel):
    id: str = Field(..., description="Unique identifier for chapter")
    name: str = Field(..., description="Chapter title")
    unit: str = Field(default="Unit 1", description="Unit or module name")
    topics: List[Topic] = Field(default_factory=list, description="Topics under this chapter")

class SubjectCurriculum(BaseModel):
    board: BoardType
    academic_year: str = Field(..., description="Academic year e.g. 2024-25")
    class_name: str = Field(..., description="Class e.g. Class 10")
    subject: str = Field(..., description="Subject name e.g. Science, Mathematics")
    chapters: List[Chapter] = Field(default_factory=list)

class BoardProfile(BaseModel):
    board: BoardType
    academic_year: str
    supported_classes: List[str]
    section_conventions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Standard section names and target question types"
    )
    marks_per_type_defaults: Dict[QuestionType, float] = Field(
        default_factory=dict,
        description="Default marks per question type for this board"
    )
    allow_internal_choice: bool = True
    language_mode: str = Field(default="ENGLISH", description="ENGLISH, HINDI, or BILINGUAL")
    pedagogical_guidelines: List[str] = Field(default_factory=list)
