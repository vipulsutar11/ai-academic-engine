import pytest
from app.models.curriculum import BoardType, SubjectCurriculum, Chapter, Topic
from app.blueprint.engine import BlueprintEngine
from app.blueprint.validator import BlueprintValidator

@pytest.fixture
def sample_curriculum():
    return SubjectCurriculum(
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        chapters=[
            Chapter(
                id="CH_01",
                name="Electricity",
                unit="Unit 1",
                topics=[
                    Topic(id="CH_01_TOPIC_01", name="Ohm's Law", weightage=1.0)
                ]
            )
        ]
    )

def test_generate_blueprint(sample_curriculum):
    bp = BlueprintEngine.generate_blueprint(
        curriculum=sample_curriculum,
        total_marks=80.0,
        duration_minutes=180
    )
    assert bp.board == BoardType.CBSE
    assert bp.total_marks > 0
    assert len(bp.sections) >= 3

def test_blueprint_question_counts(sample_curriculum):
    bp = BlueprintEngine.generate_blueprint(sample_curriculum, total_marks=20.0)
    total_q_count = sum(len(sec.question_blueprints) for sec in bp.sections)
    assert total_q_count > 0
