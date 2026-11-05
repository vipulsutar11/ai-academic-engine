import logging
from typing import List, Dict, Any, Optional
from app.models.question import Question, MarkingSchemeStep, QuestionBatch
from app.models.blueprint import QuestionBlueprint
from app.models.curriculum import BoardType, QuestionType
from app.ai.provider import BaseLLMProvider
from app.ai.prompts import PromptBuilder
from app.rag.retriever import RAGRetriever

from app.cache.question_cache import QuestionBankCache
from config import config

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Generates single questions from QuestionBlueprint and performs targeted regeneration of failed questions."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        retriever: Optional[RAGRetriever] = None
    ):
        self.provider = provider
        self.retriever = retriever
        self.question_cache = QuestionBankCache()
        self._rag_query_cache: Dict[str, List[Dict[str, Any]]] = {}

    def generate_question(
        self,
        bp: QuestionBlueprint,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        existing_questions_summary: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None,
        language_mode: str = "ENGLISH"
    ) -> Question:
        # 1. Check QuestionBankCache for instant <1ms retrieval
        if not custom_prompt:
            cached_q = self.question_cache.get_cached_question(
                chapter_id=bp.chapter_id,
                topic_id=bp.topic_id,
                difficulty=bp.difficulty.value if hasattr(bp.difficulty, "value") else str(bp.difficulty),
                question_type=bp.question_type.value if hasattr(bp.question_type, "value") else str(bp.question_type),
                subject=subject
            )
            if cached_q:
                cached_q.id = bp.id
                cached_q.marks = bp.marks
                self._adjust_marking_scheme(cached_q, bp.marks)
                return cached_q

        # 2. Retrieve context from RAG if retriever is configured (with in-memory topic caching)
        rag_context = []
        if self.retriever:
            rag_key = f"{subject}|{bp.chapter_id}|{bp.topic_id}|{custom_prompt or ''}"
            if rag_key in self._rag_query_cache:
                rag_context = self._rag_query_cache[rag_key]
            else:
                rag_query = f"{subject} {bp.chapter_id} {bp.topic_id}"
                if custom_prompt:
                    rag_query += f" {custom_prompt}"
                rag_context = self.retriever.retrieve(
                    query=rag_query,
                    board=board,
                    academic_year=academic_year,
                    class_name=class_name,
                    subject=subject,
                    chapter_id=bp.chapter_id,
                    topic_id=bp.topic_id,
                    top_k=config.TOP_K_RETRIEVAL
                )
                self._rag_query_cache[rag_key] = rag_context

        sys_prompt = PromptBuilder.build_system_prompt(board, academic_year, language_mode=language_mode)
        user_prompt = PromptBuilder.build_question_prompt(
            bp=bp,
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            rag_context=rag_context,
            existing_questions_summary=existing_questions_summary,
            custom_prompt=custom_prompt
        )

        question = self.provider.generate_structured(
            prompt=user_prompt,
            response_schema=Question,
            system_prompt=sys_prompt,
            temperature=0.3
        )

        # Enforce blueprint metadata overrides
        question.id = bp.id
        question.type = bp.question_type
        question.difficulty = bp.difficulty
        question.chapter_id = bp.chapter_id
        question.topic_id = bp.topic_id
        question.blooms_level = bp.blooms_level

        # Adjust marking scheme sum to strictly equal bp.marks
        self._adjust_marking_scheme(question, bp.marks)

        # Store in QuestionBankCache for future instant retrieval
        self.question_cache.store_question(question, subject=subject)

        logger.info(f"Generated question '{question.id}' ({question.type.value}, {question.marks} marks).")
        return question

    def generate_questions_batch(
        self,
        blueprints: List[QuestionBlueprint],
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        custom_prompt: Optional[str] = None,
        language_mode: str = "ENGLISH"
    ) -> List[Question]:
        """Generates multiple questions in a single LLM batch call for maximum throughput."""
        if not blueprints:
            return []

        results: Dict[str, Question] = {}
        pending_bps: List[QuestionBlueprint] = []

        # 1. Check cache first for instant retrieval
        for bp in blueprints:
            if not custom_prompt:
                cached_q = self.question_cache.get_cached_question(
                    chapter_id=bp.chapter_id,
                    topic_id=bp.topic_id,
                    difficulty=bp.difficulty.value if hasattr(bp.difficulty, "value") else str(bp.difficulty),
                    question_type=bp.question_type.value if hasattr(bp.question_type, "value") else str(bp.question_type),
                    subject=subject
                )
                if cached_q:
                    cached_q.id = bp.id
                    cached_q.marks = bp.marks
                    self._adjust_marking_scheme(cached_q, bp.marks)
                    results[bp.id] = cached_q
                    continue
            pending_bps.append(bp)

        if not pending_bps:
            return [results[bp.id] for bp in blueprints if bp.id in results]

        # 2. Gather RAG context for chapters in pending_bps
        rag_context = []
        if self.retriever:
            for bp in pending_bps[:3]:
                rag_key = f"{subject}|{bp.chapter_id}|{bp.topic_id}|{custom_prompt or ''}"
                if rag_key in self._rag_query_cache:
                    rag_context.extend(self._rag_query_cache[rag_key])
                else:
                    rag_query = f"{subject} {bp.chapter_id} {bp.topic_id}"
                    if custom_prompt:
                        rag_query += f" {custom_prompt}"
                    ctx = self.retriever.retrieve(
                        query=rag_query,
                        board=board,
                        academic_year=academic_year,
                        class_name=class_name,
                        subject=subject,
                        chapter_id=bp.chapter_id,
                        topic_id=bp.topic_id,
                        top_k=2
                    )
                    self._rag_query_cache[rag_key] = ctx
                    rag_context.extend(ctx)

        # 3. Call LLM to generate batch
        sys_prompt = PromptBuilder.build_system_prompt(board, academic_year, language_mode=language_mode)
        batch_prompt = PromptBuilder.build_batch_question_prompt(
            blueprints=pending_bps,
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            rag_context=rag_context,
            custom_prompt=custom_prompt
        )

        try:
            batch_result = self.provider.generate_structured(
                prompt=batch_prompt,
                response_schema=QuestionBatch,
                system_prompt=sys_prompt,
                temperature=0.3
            )
            for idx, gen_q in enumerate(batch_result.questions):
                if idx < len(pending_bps):
                    target_bp = pending_bps[idx]
                    gen_q.id = target_bp.id
                    gen_q.type = target_bp.question_type
                    gen_q.difficulty = target_bp.difficulty
                    gen_q.chapter_id = target_bp.chapter_id
                    gen_q.topic_id = target_bp.topic_id
                    gen_q.blooms_level = target_bp.blooms_level
                    self._adjust_marking_scheme(gen_q, target_bp.marks)
                    self.question_cache.store_question(gen_q, subject=subject)
                    results[target_bp.id] = gen_q
        except Exception as e:
            logger.warning(f"Batch generation failed: {e}. Falling back to individual generation.")

        # 4. Fallback for any questions that weren't generated in the batch
        for bp in pending_bps:
            if bp.id not in results:
                q = self.generate_question(
                    bp=bp,
                    board=board,
                    academic_year=academic_year,
                    class_name=class_name,
                    subject=subject,
                    custom_prompt=custom_prompt,
                    language_mode=language_mode
                )
                results[bp.id] = q

        return [results[bp.id] for bp in blueprints if bp.id in results]

    def regenerate_targeted_question(
        self,
        bp: QuestionBlueprint,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        failed_question_stem: str,
        rejection_reason: str,
        language_mode: str = "ENGLISH"
    ) -> Question:
        """Targeted regeneration of a single failed question without re-generating full paper."""
        logger.info(f"Targeted regeneration triggered for question '{bp.id}' due to reason: {rejection_reason}")

        rag_context = []
        if self.retriever:
            rag_context = self.retriever.retrieve(
                query=f"{subject} {bp.topic_id} {rejection_reason}",
                board=board,
                academic_year=academic_year,
                class_name=class_name,
                subject=subject,
                chapter_id=bp.chapter_id,
                topic_id=bp.topic_id,
                top_k=3
            )

        sys_prompt = PromptBuilder.build_system_prompt(board, academic_year, language_mode=language_mode)
        regen_prompt = PromptBuilder.build_regeneration_prompt(
            bp=bp,
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            failed_question_stem=failed_question_stem,
            rejection_reason=rejection_reason,
            rag_context=rag_context
        )

        new_question = self.provider.generate_structured(
            prompt=regen_prompt,
            response_schema=Question,
            system_prompt=sys_prompt,
            temperature=0.4
        )

        # Enforce blueprint constraints
        new_question.id = bp.id
        new_question.type = bp.question_type
        new_question.difficulty = bp.difficulty
        new_question.chapter_id = bp.chapter_id
        new_question.topic_id = bp.topic_id

        # Adjust marking scheme sum to strictly equal bp.marks
        self._adjust_marking_scheme(new_question, bp.marks)

        logger.info(f"Targeted regeneration completed for question '{new_question.id}'.")
        return new_question

    @staticmethod
    def _adjust_marking_scheme(question: Question, target_marks: float):
        question.marks = target_marks
        if not question.marking_scheme:
            question.marking_scheme = [
                MarkingSchemeStep(step_number=1, description="Correct and complete answer.", marks=target_marks)
            ]
        else:
            total_current = sum(s.marks for s in question.marking_scheme)
            if total_current <= 0:
                question.marking_scheme = [
                    MarkingSchemeStep(step_number=1, description="Correct answer.", marks=target_marks)
                ]
            else:
                ratio = target_marks / total_current
                for s in question.marking_scheme:
                    s.marks = round(s.marks * ratio, 2)
                diff = round(target_marks - sum(s.marks for s in question.marking_scheme), 2)
                question.marking_scheme[-1].marks = round(question.marking_scheme[-1].marks + diff, 2)
