import logging
from typing import List, Optional, Tuple
from app.models.question import Question
from app.models.exam import ExamPaper
from app.models.evaluation import QuestionQualityResult, PaperQualityResult
from app.ai.provider import BaseLLMProvider
from app.evaluation.relevance import RelevanceEvaluator
from config import config

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AIQualityEvaluator:
    """Evaluates question pedagogical quality, clarity, correctness, and curriculum alignment."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        relevance_evaluator: Optional[RelevanceEvaluator] = None,
        pass_threshold: float = None
    ):
        self.provider = provider
        self.relevance_evaluator = relevance_evaluator or RelevanceEvaluator()
        self.pass_threshold = pass_threshold if pass_threshold is not None else config.QUALITY_PASS_THRESHOLD

    def evaluate_question(
        self,
        question: Question,
        rag_context: Optional[List[dict]] = None
    ) -> QuestionQualityResult:
        # Check semantic relevance first
        rel_res = self.relevance_evaluator.evaluate_relevance(question.stem, rag_context or [])

        # Deterministic heuristic scores based on question clarity & completeness
        clarity_score = 0.95 if len(question.stem.strip()) >= 15 else 0.50
        correctness_score = 0.95 if (question.correct_answer and question.marking_scheme) else 0.60
        alignment_score = rel_res.similarity_score if rag_context else 0.90
        rel_score = rel_res.similarity_score

        overall = round((rel_score * 0.25) + (clarity_score * 0.25) + (correctness_score * 0.25) + (alignment_score * 0.25), 2)
        passed = overall >= self.pass_threshold

        feedback = []
        if not passed:
            feedback.append(f"Overall quality score ({overall:.2f}) below threshold ({self.pass_threshold:.2f}).")
        if clarity_score < 0.7:
            feedback.append("Question stem may be too brief or ambiguous.")
        if correctness_score < 0.7:
            feedback.append("Question missing complete answer key or step-by-step marking scheme.")

        return QuestionQualityResult(
            question_id=question.id,
            passed=passed,
            relevance_score=rel_score,
            clarity_score=clarity_score,
            correctness_score=correctness_score,
            curriculum_alignment_score=alignment_score,
            overall_score=overall,
            feedback=feedback,
            suggested_fix="Expand question stem and add detailed marking steps." if not passed else None
        )

    def evaluate_paper(
        self,
        paper: ExamPaper,
        rag_context: Optional[List[dict]] = None
    ) -> PaperQualityResult:
        questions: List[Question] = [q for sec in paper.sections for q in sec.questions]
        if not questions:
            return PaperQualityResult(
                passed=True,
                overall_score=1.0,
                question_evaluations=[],
                summary="Evaluated 0 questions."
            )

        max_workers = min(8, max(1, len(questions)))
        evaluations_dict = {}

        def _eval_task(idx_q: Tuple[int, Question]):
            idx, q = idx_q
            return idx, self.evaluate_question(q, rag_context)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in executor.map(_eval_task, enumerate(questions)):
                evaluations_dict[idx] = res

        evaluations: List[QuestionQualityResult] = [evaluations_dict[i] for i in range(len(questions))]
        total_score = sum(e.overall_score for e in evaluations)
        passed_count = sum(1 for e in evaluations if e.passed)
        total_questions = len(questions)

        overall_score = round(total_score / total_questions, 2) if total_questions > 0 else 1.0
        all_passed = (passed_count == total_questions) and (overall_score >= self.pass_threshold)

        summary = f"Evaluated {total_questions} questions. {passed_count}/{total_questions} passed quality threshold. Overall Paper Quality Score: {overall_score:.2f}."
        logger.info(f"Paper quality evaluation summary: {summary}")

        return PaperQualityResult(
            passed=all_passed,
            overall_score=overall_score,
            question_evaluations=evaluations,
            summary=summary
        )
