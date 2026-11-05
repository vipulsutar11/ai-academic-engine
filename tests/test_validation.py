import pytest
from app.models.question import Question, MCQOption, MarkingSchemeStep
from app.models.exam import ExamPaper, ExamSection
from app.models.curriculum import QuestionType, DifficultyLevel, BoardType
from app.validation.schema import SchemaValidator
from app.validation.marks import MarksValidator

def test_mcq_schema_validation():
    # Valid MCQ
    valid_mcq = Question(
        id="Q1",
        type=QuestionType.MCQ,
        stem="What is the unit of resistance?",
        options=[
            MCQOption(key="A", text="Ampere"),
            MCQOption(key="B", text="Ohm"),
            MCQOption(key="C", text="Volt"),
            MCQOption(key="D", text="Watt")
        ],
        correct_answer="B",
        marking_scheme=[MarkingSchemeStep(step_number=1, description="Option B is correct", marks=1.0)],
        marks=1.0,
        difficulty=DifficultyLevel.EASY,
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )
    passed, errors = SchemaValidator.validate_question_structure(valid_mcq)
    assert passed is True
    assert len(errors) == 0

    # Invalid MCQ (missing options)
    invalid_mcq = valid_mcq.model_copy()
    invalid_mcq.options = [MCQOption(key="A", text="Ampere")]
    passed, errors = SchemaValidator.validate_question_structure(invalid_mcq)
    assert passed is False
    assert len(errors) > 0

def test_marks_validator():
    q1 = Question(
        id="Q1",
        type=QuestionType.MCQ,
        stem="What is the unit of power?",
        options=[
            MCQOption(key="A", text="Ampere"),
            MCQOption(key="B", text="Ohm"),
            MCQOption(key="C", text="Volt"),
            MCQOption(key="D", text="Watt")
        ],
        correct_answer="D",
        marking_scheme=[MarkingSchemeStep(step_number=1, description="Option D", marks=1.0)],
        marks=1.0,
        difficulty=DifficultyLevel.EASY,
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )

    paper = ExamPaper(
        id="P1",
        title="Sample Exam",
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        total_marks=1.0,
        duration_minutes=30,
        sections=[
            ExamSection(name="Section A", instructions="MCQs", questions=[q1], total_marks=1.0)
        ]
    )

    passed, errors = MarksValidator.validate_marks(paper)
    assert passed is True
