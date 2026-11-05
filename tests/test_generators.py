import pytest
from app.ai.mock_provider import MockLLMProvider
from app.generators.mcq import MCQGenerator
from app.generators.question import QuestionGenerator
from app.generators.exam import ExamGenerator
from app.generators.assignment import AssignmentGenerator
from app.models.curriculum import BoardType, QuestionType, DifficultyLevel
from app.models.blueprint import QuestionBlueprint, ExamBlueprint, SectionBlueprint

@pytest.fixture
def mock_provider():
    return MockLLMProvider()

def test_mcq_generator(mock_provider):
    gen = MCQGenerator(provider=mock_provider)
    mcq = gen.generate_mcq(
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )
    assert mcq.type == QuestionType.MCQ
    assert len(mcq.options) == 4

def test_targeted_question_regeneration(mock_provider):
    gen = QuestionGenerator(provider=mock_provider)
    bp = QuestionBlueprint(
        id="Q_REGEN_1",
        question_type=QuestionType.SHORT_ANSWER,
        marks=3.0,
        difficulty=DifficultyLevel.MEDIUM,
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )
    new_q = gen.regenerate_targeted_question(
        bp=bp,
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        failed_question_stem="Old rejected question",
        rejection_reason="Duplicate question stem"
    )
    assert new_q.id == "Q_REGEN_1"
    assert new_q.marks == 3.0

def test_assignment_generator(mock_provider):
    gen = AssignmentGenerator(provider=mock_provider)
    assignment = gen.generate_assignment(
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        total_marks=20.0
    )
    assert assignment.total_marks == 20.0
    assert len(assignment.questions) == 3
    assert len(assignment.rubric) > 0
