import uuid
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from app.models.exam import ExamPaper, ExamSection
from app.models.blueprint import ExamBlueprint
from app.models.question import Question
from app.ai.provider import BaseLLMProvider
from app.generators.question import QuestionGenerator
from app.generators.answer import AnswerGenerator
from app.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

class ExamGenerator:
    """Generates complete Exam Papers driven strictly by ExamBlueprint specifications."""

    def __init__(self, provider: BaseLLMProvider, retriever: Optional[RAGRetriever] = None):
        self.question_generator = QuestionGenerator(provider=provider, retriever=retriever)

    def generate_exam_paper(self, blueprint: ExamBlueprint, language_mode: str = "ENGLISH") -> ExamPaper:
        paper_id = f"EXAM_{uuid.uuid4().hex[:8]}"
        sections: List[ExamSection] = []

        logger.info(f"Generating exam paper ({language_mode}) from blueprint '{blueprint.title}' with {len(blueprint.sections)} sections in parallel.")

        batch_tasks = []
        for sec_idx, sec_bp in enumerate(blueprint.sections):
            bps = sec_bp.question_blueprints
            chunk_size = 10
            for i in range(0, len(bps), chunk_size):
                chunk = bps[i:i + chunk_size]
                batch_tasks.append((sec_idx, i, chunk))

        def _gen_batch_task(task):
            sec_idx, start_idx, chunk_bps = task
            questions = self.question_generator.generate_questions_batch(
                blueprints=chunk_bps,
                board=blueprint.board,
                academic_year=blueprint.academic_year,
                class_name=blueprint.class_name,
                subject=blueprint.subject,
                language_mode=language_mode
            )
            return sec_idx, start_idx, questions

        results = {}
        max_workers = min(4, max(1, len(batch_tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sec_idx, start_idx, questions in executor.map(_gen_batch_task, batch_tasks):
                if sec_idx not in results:
                    results[sec_idx] = {}
                for offset, q in enumerate(questions):
                    results[sec_idx][start_idx + offset] = q

        for sec_idx, sec_bp in enumerate(blueprint.sections):
            section_questions: List[Question] = []
            sec_dict = results.get(sec_idx, {})
            for q_idx in range(len(sec_bp.question_blueprints)):
                if q_idx in sec_dict:
                    section_questions.append(sec_dict[q_idx])

            sec = ExamSection(
                name=sec_bp.name,
                instructions=sec_bp.instructions,
                questions=section_questions,
                total_marks=sec_bp.total_marks
            )
            sections.append(sec)

        paper = ExamPaper(
            id=paper_id,
            title=blueprint.title,
            board=blueprint.board,
            academic_year=blueprint.academic_year,
            class_name=blueprint.class_name,
            subject=blueprint.subject,
            total_marks=blueprint.total_marks,
            duration_minutes=blueprint.duration_minutes,
            general_instructions=[
                "1. All questions are compulsory unless internal choice is explicitly stated.",
                "2. Section A contains 1-mark objective questions.",
                "3. Write concise, clear answers with step-by-step working for numerical problems.",
                "4. Wherever applicable, draw neat and labeled diagrams."
            ],
            sections=sections,
            answer_key=[]
        )

        paper.answer_key = AnswerGenerator.generate_answer_keys_for_paper(paper)
        logger.info(f"Successfully generated full exam paper '{paper.title}' (ID: {paper.id}).")
        return paper
