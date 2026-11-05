import pytest
from app.models.question import Question, MCQOption, MarkingSchemeStep
from app.models.curriculum import QuestionType, DifficultyLevel
from app.validation.duplicate import SemanticDuplicateDetector

def test_semantic_duplicate_detection():
    detector = SemanticDuplicateDetector(threshold=0.80)

    q1 = Question(
        id="Q1",
        type=QuestionType.SHORT_ANSWER,
        stem="State Ohm's Law and write its mathematical formula V = I * R.",
        correct_answer="V = I * R",
        marking_scheme=[MarkingSchemeStep(step_number=1, description="Definition", marks=2.0)],
        marks=2.0,
        difficulty=DifficultyLevel.EASY,
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )

    q2 = Question(
        id="Q2",
        type=QuestionType.SHORT_ANSWER,
        stem="State Ohm's Law and write its mathematical formula V = I * R in detail.",
        correct_answer="V = I * R",
        marking_scheme=[MarkingSchemeStep(step_number=1, description="Definition", marks=2.0)],
        marks=2.0,
        difficulty=DifficultyLevel.EASY,
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )

    q_diff = Question(
        id="Q3",
        type=QuestionType.SHORT_ANSWER,
        stem="Explain the mechanism of photosynthesis in green leaves.",
        correct_answer="Chlorophyll absorbs sunlight...",
        marking_scheme=[MarkingSchemeStep(step_number=1, description="Explanation", marks=2.0)],
        marks=2.0,
        difficulty=DifficultyLevel.MEDIUM,
        chapter_id="CH_02",
        topic_id="CH_02_TOPIC_01"
    )

    is_dup, sim, matched = detector.is_duplicate(q2, [q1])
    assert is_dup is True
    assert sim >= 0.80

    is_dup_diff, sim_diff, _ = detector.is_duplicate(q_diff, [q1])
    assert is_dup_diff is False
