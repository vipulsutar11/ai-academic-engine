import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple
from app.models.curriculum import BoardType, SubjectCurriculum, Chapter, Topic
from app.models.blueprint import ExamBlueprint
from app.models.exam import ExamPaper
from app.models.assignment import Assignment
from app.models.question import Question
from app.models.evaluation import PaperQualityResult
from app.models.api_models import ExamGenerationRequest
from app.curriculum.board_profiles import BoardProfileRegistry
from app.curriculum.loader import CurriculumLoader
from app.blueprint.engine import BlueprintEngine
from app.rag.retriever import RAGRetriever
from app.rag.chunker import CurriculumChunker
from app.ai.provider import BaseLLMProvider
from app.ai.mistral import MistralProvider
from app.ai.gemini import GeminiProvider
from app.ai.mock_provider import MockLLMProvider
from app.ai.fallback import TwoTierFallbackProvider
from app.generators.exam import ExamGenerator
from app.generators.question import QuestionGenerator
from app.generators.assignment import AssignmentGenerator
from app.generators.answer import AnswerGenerator
from app.validation.paper import DeterministicPaperValidator
from app.validation.duplicate import SemanticDuplicateDetector
from app.evaluation.quality import AIQualityEvaluator
from config import config

logger = logging.getLogger(__name__)

class AcademicContentPipeline:
    """Master Pipeline orchestrating Board Profile loading, Blueprint Generation, RAG Context Retrieval,
    LLM Generation, Deterministic Validation, AI Quality Evaluation, Semantic Duplicate Detection,
    and Targeted Regeneration.
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        rag_retriever: Optional[RAGRetriever] = None
    ):
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            def _is_real_key(key: Optional[str]) -> bool:
                if not key:
                    return False
                k = key.strip().lower()
                return not (k.startswith("mock") or k.startswith("your_") or "here" in k or "<" in k)

            # Primary: Mistral AI, Fallback: Google Gemini, Emergency: Mock Engine
            primary = MistralProvider() if _is_real_key(config.MISTRAL_API_KEY) else MockLLMProvider()
            fallback = GeminiProvider() if _is_real_key(config.GEMINI_API_KEY) else MockLLMProvider()

            self.llm_provider = TwoTierFallbackProvider(
                primary_provider=primary,
                secondary_provider=fallback,
                tertiary_provider=MockLLMProvider()
            )

        self.rag_retriever = rag_retriever or RAGRetriever()
        self.exam_generator = ExamGenerator(provider=self.llm_provider, retriever=self.rag_retriever)
        self.question_generator = QuestionGenerator(provider=self.llm_provider, retriever=self.rag_retriever)
        self.assignment_generator = AssignmentGenerator(provider=self.llm_provider, retriever=self.rag_retriever)
        
        self.duplicate_detector = SemanticDuplicateDetector()
        self.paper_validator = DeterministicPaperValidator(duplicate_detector=self.duplicate_detector)
        self.quality_evaluator = AIQualityEvaluator(provider=self.llm_provider)
        self._indexed_curriculums = set()

    def generate_validated_exam_paper(
        self,
        request: ExamGenerationRequest,
        curriculum: Optional[SubjectCurriculum] = None,
        max_regeneration_attempts: int = 1
    ) -> Tuple[ExamPaper, PaperQualityResult, List[str]]:
        # Step 1: Infer metadata from prompt if user only provided a prompt
        p_lower = (getattr(request, "prompt", None) or "").lower()

        inferred_subject = request.subject
        if not inferred_subject or inferred_subject == "Science":
            if "history" in p_lower:
                inferred_subject = "History"
            elif "math" in p_lower or "mathematics" in p_lower:
                inferred_subject = "Mathematics"
            elif "physics" in p_lower or "chemistry" in p_lower or "biology" in p_lower or "science" in p_lower:
                inferred_subject = "Science"
            elif "english" in p_lower:
                inferred_subject = "English"
            elif not inferred_subject:
                inferred_subject = "Science"

        inferred_board = request.board or BoardType.CBSE
        if not request.board:
            if "icse" in p_lower:
                inferred_board = BoardType.ICSE
            elif "up board" in p_lower or "upboard" in p_lower:
                inferred_board = BoardType.UP_BOARD

        inferred_class = request.class_name or "Class 10"
        class_match = re.search(r"class\s*([0-9]+)", p_lower)
        if class_match:
            inferred_class = f"Class {class_match.group(1)}"

        # Step 2: Load Board Profile and active language mode
        profile = BoardProfileRegistry.get_profile(inferred_board, request.academic_year)
        lang_mode = request.language_mode if getattr(request, "language_mode", None) and request.language_mode != "ENGLISH" else profile.language_mode
        if "bilingual" in p_lower:
            lang_mode = "BILINGUAL"
        elif "hindi" in p_lower:
            lang_mode = "HINDI"
        logger.info(f"Loaded Board Profile for {profile.board.value} ({profile.academic_year}) with active language mode: {lang_mode}.")

        # Step 3: Build / Resolve Subject Curriculum
        curriculum = curriculum or CurriculumLoader.find_and_load_curriculum(
            board=inferred_board,
            academic_year=request.academic_year,
            class_name=inferred_class,
            subject=inferred_subject
        ) or self._build_default_curriculum(
            board=inferred_board,
            academic_year=request.academic_year,
            class_name=inferred_class,
            subject=inferred_subject
        )

        curr_key = (inferred_board.value if hasattr(inferred_board, "value") else str(inferred_board), request.academic_year, inferred_class, inferred_subject)
        if curr_key not in self._indexed_curriculums:
            chunks = CurriculumChunker.chunk_curriculum(curriculum)
            raw_chunks = [{"text": c.text, "metadata": c.metadata} for c in chunks]
            self.rag_retriever.index_curriculum_chunks(raw_chunks)
            self._indexed_curriculums.add(curr_key)

        # Step 3: Apply Blueprint Engine
        blueprint = request.custom_blueprint or BlueprintEngine.generate_blueprint(
            curriculum=curriculum,
            total_marks=request.total_marks,
            duration_minutes=request.duration_minutes,
            title=request.title,
            prompt=request.prompt
        )
        logger.info(f"Blueprint generated: Total Marks = {blueprint.total_marks}, Sections = {len(blueprint.sections)}.")

        # Step 4: Initial Paper Generation
        paper = self.exam_generator.generate_exam_paper(blueprint, language_mode=lang_mode)

        # Step 5 & 6 & 7 & 8: Validation, Quality Evaluation, Duplicate Check & Targeted Regeneration Loop
        attempts = 0
        all_warnings: List[str] = []

        while attempts < max_regeneration_attempts:
            attempts += 1
            logger.info(f"Pipeline verification pass {attempts}/{max_regeneration_attempts}...")

            # Deterministic Validation
            val_passed, val_errors = self.paper_validator.validate_paper(
                paper=paper,
                blueprint=blueprint,
                curriculum=curriculum
            )

            # AI Quality Evaluation
            quality_result = self.quality_evaluator.evaluate_paper(paper)

            if val_passed and quality_result.passed:
                logger.info(f"Exam Paper '{paper.title}' successfully passed all deterministic and quality validations on pass {attempts}!")
                break

            # If validation or quality evaluation failed, identify targeted questions to regenerate
            logger.warning(f"Pass {attempts} flagged issues. Running targeted question regeneration...")

            failed_q_ids = set()
            for err in val_errors:
                all_warnings.append(f"Pass {attempts} error: {err}")
                # Parse question ID if mentioned
                for sec in paper.sections:
                    for q in sec.questions:
                        if q.id in err:
                            failed_q_ids.add(q.id)

            for q_eval in quality_result.question_evaluations:
                if not q_eval.passed:
                    failed_q_ids.add(q_eval.question_id)

            if not failed_q_ids:
                # Fallback if specific ID not isolated: regenerate last question of first failing section
                if paper.sections and paper.sections[0].questions:
                    failed_q_ids.add(paper.sections[0].questions[0].id)

            # Concurrent targeted regeneration of ONLY failed questions
            regen_tasks = []
            for sec_idx, sec in enumerate(paper.sections):
                for q_idx, q in enumerate(sec.questions):
                    if q.id in failed_q_ids:
                        bp_q = self._find_question_blueprint(blueprint, q.id)
                        if bp_q:
                            reasons = "; ".join(val_errors[:2]) if val_errors else "Quality score below threshold"
                            regen_tasks.append((sec_idx, q_idx, bp_q, q.stem, reasons))

            if regen_tasks:
                def _exec_regen(task):
                    sec_i, q_i, bp_q, failed_stem, reasons = task
                    new_q = self.question_generator.regenerate_targeted_question(
                        bp=bp_q,
                        board=paper.board,
                        academic_year=paper.academic_year,
                        class_name=paper.class_name,
                        subject=paper.subject,
                        failed_question_stem=failed_stem,
                        rejection_reason=reasons,
                        language_mode=lang_mode
                    )
                    return sec_i, q_i, new_q

                max_regen_workers = min(6, len(regen_tasks))
                with ThreadPoolExecutor(max_workers=max_regen_workers) as executor:
                    for sec_i, q_i, new_q in executor.map(_exec_regen, regen_tasks):
                        paper.sections[sec_i].questions[q_i] = new_q
                        logger.info(f"Targeted regeneration concurrently replaced question '{new_q.id}'.")

            # Update Answer Keys
            paper.answer_key = AnswerGenerator.generate_answer_keys_for_paper(paper)

        # Final quality evaluation
        final_quality = self.quality_evaluator.evaluate_paper(paper)
        logger.info(f"Pipeline finished. Final paper total marks = {paper.total_marks}, Quality score = {final_quality.overall_score}.")

        return paper, final_quality, all_warnings

    def generate_batch_exam_papers(
        self,
        requests: List[ExamGenerationRequest],
        max_workers: int = 10
    ) -> List[Tuple[ExamPaper, PaperQualityResult, List[str]]]:
        """Generates multiple exam papers concurrently in parallel using worker pool threads.
        Pre-seeds curriculum index caches to eliminate redundant chunking and RAG overhead.
        """
        logger.info(f"Starting batch generation of {len(requests)} exam papers with max_workers={max_workers}...")

        # Pre-seed curriculum chunks for unique subjects/classes
        for req in requests:
            curr_key = (req.board.value if hasattr(req.board, "value") else str(req.board), req.academic_year, req.class_name, req.subject)
            if curr_key not in self._indexed_curriculums:
                curriculum = CurriculumLoader.find_and_load_curriculum(
                    board=req.board,
                    academic_year=req.academic_year,
                    class_name=req.class_name,
                    subject=req.subject
                ) or self._build_default_curriculum(
                    board=req.board,
                    academic_year=req.academic_year,
                    class_name=req.class_name,
                    subject=req.subject
                )
                chunks = CurriculumChunker.chunk_curriculum(curriculum)
                raw_chunks = [{"text": c.text, "metadata": c.metadata} for c in chunks]
                self.rag_retriever.index_curriculum_chunks(raw_chunks)
                self._indexed_curriculums.add(curr_key)

        from concurrent.futures import ThreadPoolExecutor

        workers = min(max_workers, max(1, len(requests)))
        results_dict = {}

        def _gen_single(req_tuple):
            idx, req = req_tuple
            return idx, self.generate_validated_exam_paper(req)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, res in executor.map(_gen_single, enumerate(requests)):
                results_dict[idx] = res

        ordered_results = [results_dict[i] for i in range(len(requests))]
        logger.info(f"Completed batch generation of {len(requests)} exam papers.")
        return ordered_results

    @staticmethod
    def _find_question_blueprint(blueprint: ExamBlueprint, q_id: str):
        for sec in blueprint.sections:
            for q_bp in sec.question_blueprints:
                if q_bp.id == q_id:
                    return q_bp
        return None

    @staticmethod
    def _build_default_curriculum(board: BoardType, academic_year: str, class_name: str, subject: str) -> SubjectCurriculum:
        """Fallback curriculum builder with subject-appropriate default chapters."""
        subj_lower = subject.lower()
        if "math" in subj_lower:
            ch1 = Chapter(
                id="CH_MATH_01",
                name="Real Numbers, Polynomials & Equations",
                unit="Unit 1",
                topics=[
                    Topic(id="CH_MATH_01_TOPIC_01", name="Real Numbers & Fundamental Theorem of Arithmetic", weightage=1.0, keywords=["real numbers", "prime factorization", "hcf", "lcm"]),
                    Topic(id="CH_MATH_01_TOPIC_02", name="Polynomials and Quadratic Equations", weightage=1.0, keywords=["polynomials", "zeroes", "quadratic", "roots"])
                ]
            )
            ch2 = Chapter(
                id="CH_MATH_02",
                name="Trigonometry, Coordinate Geometry & Triangles",
                unit="Unit 2",
                topics=[
                    Topic(id="CH_MATH_02_TOPIC_01", name="Introduction to Trigonometry & Identities", weightage=1.0, keywords=["trigonometry", "sin", "cos", "tan", "identities"]),
                    Topic(id="CH_MATH_02_TOPIC_02", name="Coordinate Geometry & Similar Triangles", weightage=1.0, keywords=["distance formula", "section formula", "triangles", "similarity"])
                ]
            )
            return SubjectCurriculum(
                board=board,
                academic_year=academic_year,
                class_name=class_name,
                subject=subject,
                chapters=[ch1, ch2]
            )

        if "hist" in subj_lower or "social" in subj_lower:
            ch1 = Chapter(
                id="CH_HIST_01",
                name="The Rise of Nationalism in Europe",
                unit="Unit 1: India and the Contemporary World - II",
                topics=[
                    Topic(id="CH_HIST_01_TOPIC_01", name="The French Revolution and the Idea of the Nation", weightage=1.0, keywords=["french revolution", "civil code 1804", "la patrie", "nationalism"]),
                    Topic(id="CH_HIST_01_TOPIC_02", name="The Making of Nationalism in Europe and Liberal Nationalism", weightage=1.0, keywords=["liberal nationalism", "zollverein", "treaty of vienna 1815", "mazzini"])
                ]
            )
            ch2 = Chapter(
                id="CH_HIST_02",
                name="Nationalism in India",
                unit="Unit 1: India and the Contemporary World - II",
                topics=[
                    Topic(id="CH_HIST_02_TOPIC_01", name="First World War, Khilafat and Satyagraha", weightage=1.0, keywords=["satyagraha", "rowlatt act", "jallianwala bagh", "khilafat"]),
                    Topic(id="CH_HIST_02_TOPIC_02", name="Non-Cooperation Movement and Differing Strands", weightage=1.0, keywords=["non-cooperation", "boycott", "chauri chaura", "swaraj"])
                ]
            )
            ch3 = Chapter(
                id="CH_HIST_03",
                name="The Making of a Global World & Print Culture",
                unit="Unit 2: India and the Contemporary World - II",
                topics=[
                    Topic(id="CH_HIST_03_TOPIC_01", name="The Pre-modern World and Great Depression", weightage=1.0, keywords=["silk routes", "indentured labour", "great depression 1929"]),
                    Topic(id="CH_HIST_03_TOPIC_02", name="Print Culture and Religious Reform in India", weightage=1.0, keywords=["gutenberg", "print revolution", "vernacular press act 1878"])
                ]
            )
            return SubjectCurriculum(
                board=board,
                academic_year=academic_year,
                class_name=class_name,
                subject=subject,
                chapters=[ch1, ch2, ch3]
            )

        if "sci" in subj_lower or "phys" in subj_lower or "chem" in subj_lower or "bio" in subj_lower:
            ch1 = Chapter(
                id="CH_SCI_01",
                name="Chemical Reactions and Electricity",
                unit="Unit 1",
                topics=[
                    Topic(id="CH_SCI_01_TOPIC_01", name="Chemical Equations, Oxidation and Reduction", weightage=1.0, keywords=["chemical equation", "balancing", "redox", "oxidation"]),
                    Topic(id="CH_SCI_01_TOPIC_02", name="Electric Current, Ohm's Law and Resistance", weightage=1.0, keywords=["electric current", "potential difference", "ohm's law", "resistance"])
                ]
            )
            ch2 = Chapter(
                id="CH_SCI_02",
                name="Light - Reflection and Refraction & Life Processes",
                unit="Unit 2",
                topics=[
                    Topic(id="CH_SCI_02_TOPIC_01", name="Spherical Mirrors, Lenses and Refraction", weightage=1.0, keywords=["mirror", "lens", "focal length", "refraction"]),
                    Topic(id="CH_SCI_02_TOPIC_02", name="Life Processes: Nutrition and Respiration", weightage=1.0, keywords=["nutrition", "photosynthesis", "respiration", "circulation"])
                ]
            )
            return SubjectCurriculum(
                board=board,
                academic_year=academic_year,
                class_name=class_name,
                subject=subject,
                chapters=[ch1, ch2]
            )

        # Dynamic fallback for any other subject (e.g. English, Hindi, Geography, etc.)
        ch1 = Chapter(
            id=f"CH_{subject.upper()[:4]}_01",
            name=f"{subject} Core Concepts & Foundations",
            unit="Unit 1",
            topics=[
                Topic(id=f"CH_{subject.upper()[:4]}_01_TOPIC_01", name=f"{subject} Fundamental Principles", weightage=1.0, keywords=[subject.lower(), "principles", "concepts"]),
                Topic(id=f"CH_{subject.upper()[:4]}_01_TOPIC_02", name=f"{subject} Key Themes and Analysis", weightage=1.0, keywords=[subject.lower(), "analysis", "themes"])
            ]
        )
        ch2 = Chapter(
            id=f"CH_{subject.upper()[:4]}_02",
            name=f"{subject} Advanced Applications & Practical Studies",
            unit="Unit 2",
            topics=[
                Topic(id=f"CH_{subject.upper()[:4]}_02_TOPIC_01", name=f"{subject} Critical Perspectives", weightage=1.0, keywords=[subject.lower(), "critical", "applications"]),
                Topic(id=f"CH_{subject.upper()[:4]}_02_TOPIC_02", name=f"{subject} Synthesis and Evaluation", weightage=1.0, keywords=[subject.lower(), "evaluation", "synthesis"])
            ]
        )
        return SubjectCurriculum(
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            chapters=[ch1, ch2]
        )
