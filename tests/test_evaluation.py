import pytest
from app.models.question import Question, MCQOption, MarkingSchemeStep
from app.models.curriculum import QuestionType, DifficultyLevel
from app.evaluation.quality import AIQualityEvaluator
from app.evaluation.relevance import RelevanceEvaluator

def test_relevance_evaluator():
    evaluator = RelevanceEvaluator()
    rag_context = [
        {"text": "Ohm's Law states that V = IR where V is potential difference and I is current."}
    ]
    res = evaluator.evaluate_relevance(
        question_text="State Ohm's Law and its mathematical formula.",
        rag_context=rag_context
    )
    assert res.is_relevant is True
    assert res.similarity_score > 0.30

def test_ai_quality_evaluator():
    evaluator = AIQualityEvaluator(pass_threshold=0.70)
    q = Question(
        id="Q1",
        type=QuestionType.SHORT_ANSWER,
        stem="State Ohm's law and derive the relation between voltage, current and resistance in a circuit.",
        correct_answer="V = IR",
        marking_scheme=[MarkingSchemeStep(step_number=1, description="Statement and formula V=IR", marks=3.0)],
        marks=3.0,
        difficulty=DifficultyLevel.MEDIUM,
        chapter_id="CH_01",
        topic_id="CH_01_TOPIC_01"
    )

    res = evaluator.evaluate_question(q)
    assert res.passed is True
    assert res.overall_score >= 0.70
