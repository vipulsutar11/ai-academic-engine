from typing import List, Optional
from pydantic import BaseModel, Field

class QuestionQualityResult(BaseModel):
    question_id: str
    passed: bool
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    clarity_score: float = Field(..., ge=0.0, le=1.0)
    correctness_score: float = Field(..., ge=0.0, le=1.0)
    curriculum_alignment_score: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    feedback: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None

class PaperQualityResult(BaseModel):
    passed: bool
    overall_score: float = Field(..., ge=0.0, le=1.0)
    question_evaluations: List[QuestionQualityResult] = Field(default_factory=list)
    summary: str = Field(..., description="Summary evaluation description")

class RelevanceResult(BaseModel):
    is_relevant: bool
    similarity_score: float
    matched_chunks: List[str] = Field(default_factory=list)
