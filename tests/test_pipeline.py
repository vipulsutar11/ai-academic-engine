import pytest
from app.models.api_models import ExamGenerationRequest
from app.models.curriculum import BoardType
from app.pipeline import AcademicContentPipeline
from app.ai.mock_provider import MockLLMProvider

def test_full_academic_content_pipeline():
    mock_provider = MockLLMProvider()
    pipeline = AcademicContentPipeline(llm_provider=mock_provider)

    req = ExamGenerationRequest(
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        total_marks=10.0,
        duration_minutes=60,
        title="Class 10 Science Unit Test"
    )

    paper, quality, warnings = pipeline.generate_validated_exam_paper(req)

    assert paper.board == BoardType.CBSE
    assert paper.total_marks == 10.0
    assert len(paper.sections) > 0
    assert quality.overall_score > 0.0
    assert len(paper.answer_key) > 0
