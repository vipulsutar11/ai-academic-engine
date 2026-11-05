import re
import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from app.ai.provider import BaseLLMProvider
from app.models.question import Question, MCQOption, MarkingSchemeStep
from app.models.exam import ExamPaper, ExamSection
from app.models.assignment import Assignment, AssignmentRubricItem
from app.models.evaluation import QuestionQualityResult, PaperQualityResult
from app.models.curriculum import QuestionType, DifficultyLevel, BoardType

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class MockLLMProvider(BaseLLMProvider):
    """High-quality deterministic Mock Provider for offline execution, testing, and CI/CD."""

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> T:
        logger.info(f"MockLLMProvider generating mock response for schema {response_schema.__name__}")

        schema_name = response_schema.__name__

        if schema_name == "Question":
            return self._build_mock_question_from_prompt(prompt)

        elif schema_name == "ExamPaper":
            return self._build_mock_exam_paper()

        elif schema_name == "Assignment":
            return self._build_mock_assignment()

        elif schema_name == "QuestionQualityResult":
            return QuestionQualityResult(
                question_id="Q_MOCK_1",
                passed=True,
                relevance_score=0.95,
                clarity_score=0.92,
                correctness_score=0.98,
                curriculum_alignment_score=0.96,
                overall_score=0.95,
                feedback=["High quality question matching curriculum guidelines."]
            )

        elif schema_name == "PaperQualityResult":
            return PaperQualityResult(
                passed=True,
                overall_score=0.94,
                question_evaluations=[],
                summary="Exam paper strictly satisfies all pedagogical and blueprint constraints."
            )

        # Generic fallback for any other Pydantic schema
        return response_schema.model_construct()

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        return "Mock generated response covering academic curriculum criteria."

    def _build_mock_question_from_prompt(self, prompt: str) -> Question:
        # Extract question ID from prompt if present
        q_id_match = re.search(r"Q_[A-Za-z0-9_]+", prompt)
        q_id = q_id_match.group(0) if q_id_match else "Q_MOCK_1"

        # Determine type
        if "MCQ" in prompt or "multiple choice" in prompt.lower():
            q_type = QuestionType.MCQ
        elif "CASE_STUDY" in prompt or "case-based" in prompt.lower():
            q_type = QuestionType.CASE_STUDY
        elif "LONG_ANSWER" in prompt or "5.0" in prompt or "6.0" in prompt:
            q_type = QuestionType.LONG_ANSWER
        else:
            q_type = QuestionType.SHORT_ANSWER

        # Parse marks if available in prompt
        marks_match = re.search(r"Marks Allocated:\s*([0-9.]+)", prompt)
        marks = float(marks_match.group(1)) if marks_match else (1.0 if q_type == QuestionType.MCQ else 3.0)

        # Vary stem based on q_id or hash to avoid semantic duplicate detection flags across paper
        if q_type == QuestionType.MCQ:
            if "2" in q_id:
                stem = f"Which SI unit is used to measure electric current in a closed circuit? ({q_id})"
                options = [
                    MCQOption(key="A", text="Ampere (A)", explanation="SI unit of current"),
                    MCQOption(key="B", text="Volt (V)", explanation="Unit of potential difference"),
                    MCQOption(key="C", text="Ohm (Ω)", explanation="Unit of resistance"),
                    MCQOption(key="D", text="Watt (W)", explanation="Unit of power")
                ]
                correct = "A"
            else:
                stem = f"Which SI unit is used to measure electrical potential difference across a conductor? ({q_id})"
                options = [
                    MCQOption(key="A", text="Ampere (A)", explanation="Unit of electric current"),
                    MCQOption(key="B", text="Volt (V)", explanation="Unit of potential difference"),
                    MCQOption(key="C", text="Ohm (Ω)", explanation="Unit of electrical resistance"),
                    MCQOption(key="D", text="Watt (W)", explanation="Unit of electrical power")
                ]
                correct = "B"

            return Question(
                id=q_id,
                type=QuestionType.MCQ,
                stem=stem,
                options=options,
                correct_answer=correct,
                marking_scheme=[MarkingSchemeStep(step_number=1, description=f"Correct option {correct}", marks=marks)],
                marks=marks,
                difficulty=DifficultyLevel.EASY,
                chapter_id="CH_01",
                topic_id="CH_01_TOPIC_01",
                explanation="Standard SI unit definition.",
                blooms_level="Knowledge"
            )

        elif q_type == QuestionType.CASE_STUDY:
            return Question(
                id=q_id,
                type=QuestionType.CASE_STUDY,
                stem=f"Case Study ({q_id}): An electrician installs a fuse wire rated at 5A in a circuit connected to a 220V AC supply with appliances totaling 1500W. Analyze the circuit safety and fuse behavior.",
                options=None,
                correct_answer="Current I = P/V = 1500W / 220V = 6.81A. Since current exceeds fuse rating (5A), the fuse wire melts and disconnects circuit.",
                marking_scheme=[
                    MarkingSchemeStep(step_number=1, description="Calculate total current I = P/V.", marks=round(marks * 0.5, 2)),
                    MarkingSchemeStep(step_number=2, description="Compare current with fuse rating and conclude fuse melts.", marks=round(marks * 0.5, 2))
                ],
                marks=marks,
                difficulty=DifficultyLevel.HARD,
                chapter_id="CH_01",
                topic_id="CH_01_TOPIC_02",
                explanation="Uses P = V * I calculation and safety cutoff logic.",
                blooms_level="Application"
            )

        elif q_type == QuestionType.LONG_ANSWER:
            return Question(
                id=q_id,
                type=QuestionType.LONG_ANSWER,
                stem=f"Derive Joule's Heating Effect formula H = I^2 R t for an electrical conductor ({q_id}). Calculate heat produced when 10A current flows through 20 Ohm resistor for 2 minutes.",
                options=None,
                correct_answer="Derivation: Work W = V * Q = V * I * t. By Ohm's law V = I * R, so H = I^2 * R * t. Calculation: H = (10)^2 * 20 * 120 = 240,000 Joules = 240 kJ.",
                marking_scheme=[
                    MarkingSchemeStep(step_number=1, description="Derive formula H = I^2 R t step by step.", marks=round(marks * 0.5, 2)),
                    MarkingSchemeStep(step_number=2, description="Substitute values and solve numerical problem H = 240 kJ.", marks=round(marks * 0.5, 2))
                ],
                marks=marks,
                difficulty=DifficultyLevel.HARD,
                chapter_id="CH_01",
                topic_id="CH_01_TOPIC_02",
                explanation="Detailed derivation and numerical calculation.",
                blooms_level="Analysis"
            )

        else:
            # Short answer variations
            if "Section_C" in q_id or "CH_02" in prompt:
                stem = f"Explain factors affecting electrical resistance of a cylindrical conductor ({q_id})."
                ans = "Resistance R is directly proportional to length (L) and inversely proportional to cross-sectional area (A)."
            else:
                stem = f"State Ohm's Law and write mathematical formula relating V, I, and R ({q_id})."
                ans = "Current I is proportional to potential difference V at constant temperature. V = I * R."

            return Question(
                id=q_id,
                type=QuestionType.SHORT_ANSWER,
                stem=stem,
                options=None,
                correct_answer=ans,
                marking_scheme=[
                    MarkingSchemeStep(step_number=1, description="State physical law/principle clearly.", marks=round(marks * 0.5, 2)),
                    MarkingSchemeStep(step_number=2, description="Write correct formula and unit definitions.", marks=round(marks * 0.5, 2))
                ],
                marks=marks,
                difficulty=DifficultyLevel.MEDIUM,
                chapter_id="CH_01",
                topic_id="CH_01_TOPIC_01",
                explanation="Requires definition and mathematical formula.",
                blooms_level="Understanding"
            )

    def _build_mock_exam_paper(self) -> ExamPaper:
        q1 = self._build_mock_question_from_prompt("Q_Section_A_1 MCQ 1.0")
        q2 = self._build_mock_question_from_prompt("Q_Section_B_1 SHORT_ANSWER 3.0")

        return ExamPaper(
            id="EXAM_MOCK_101",
            title="Mock CBSE Class 10 Science Examination",
            board=BoardType.CBSE,
            academic_year="2024-25",
            class_name="Class 10",
            subject="Science",
            total_marks=4.0,
            duration_minutes=60,
            general_instructions=[
                "Read all questions carefully before answering.",
                "All questions are compulsory."
            ],
            sections=[
                ExamSection(name="Section A", instructions="Objective Questions", questions=[q1], total_marks=1.0),
                ExamSection(name="Section B", instructions="Short Answer Questions", questions=[q2], total_marks=3.0)
            ]
        )

    def _build_mock_assignment(self) -> Assignment:
        q1 = self._build_mock_question_from_prompt("Q_ASSIGN_1 SHORT_ANSWER 5.0")
        return Assignment(
            id="ASSIGN_MOCK_1",
            title="Class 10 Science Practice Assignment",
            board=BoardType.CBSE,
            academic_year="2024-25",
            class_name="Class 10",
            subject="Science",
            total_marks=5.0,
            instructions=["Complete all numerical problems showing full working."],
            questions=[q1],
            rubric=[
                AssignmentRubricItem(criterion="Concept Accuracy", description="Demonstrates clear grasp of physical principles", max_marks=3.0),
                AssignmentRubricItem(criterion="Working & Steps", description="Step-by-step mathematical calculation", max_marks=2.0)
            ]
        )
