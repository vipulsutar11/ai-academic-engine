import logging
from typing import List, Dict, Any, Optional
from app.models.curriculum import BoardType, QuestionType, DifficultyLevel
from app.models.blueprint import QuestionBlueprint, ExamBlueprint

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Prompt template engine tailored for CBSE, ICSE, and UP Board curricula requirements."""

    @staticmethod
    def build_system_prompt(board: BoardType, academic_year: str, language_mode: str = "ENGLISH") -> str:
        base_prompt = (
            f"You are a master academic content developer and curriculum specialist for {board.value} ({academic_year}).\n"
            "Your output must adhere strictly to pedagogical guidelines, Bloom's Taxonomy, and official marking patterns.\n"
            "Ensure factual accuracy, correct numerical values, precise scientific units, and unambiguous option choices."
        )

        lang_upper = language_mode.upper() if language_mode else "ENGLISH"
        if lang_upper == "HINDI":
            base_prompt += (
                "\n\nCRITICAL LANGUAGE MANDATE (HINDI MEDIUM):\n"
                "- Write ALL question stems, options, answers, step-by-step marking schemes, and explanations strictly in Devanagari Hindi script (हिन्दी माध्यम).\n"
                "- Use standard NCERT Hindi medium terminology e.g., 'ओम का नियम (Ohm's Law)', 'प्रकाश-संश्लेषण (Photosynthesis)'."
            )
        elif lang_upper == "BILINGUAL" or board == BoardType.UP_BOARD:
            base_prompt += (
                "\n\nCRITICAL LANGUAGE MANDATE (BILINGUAL MEDIUM):\n"
                "- Provide main technical terms, definitions, questions, and marking steps in Devanagari Hindi with English translations in parentheses.\n"
                "- Example: 'प्रतिरोध (Resistance) का मात्रक क्या है?'"
            )
        elif board == BoardType.ICSE:
            base_prompt += (
                "\nConstraint for ICSE Board:\n"
                "- Emphasize formal definitions, exact SI units, diagrammatic labels, and step-by-step mathematical working."
            )
        elif board == BoardType.CBSE:
            base_prompt += (
                "\nConstraint for CBSE Board:\n"
                "- Follow NCERT guidelines, NCERT exemplar problem styles, and competency-based assessment objectives."
            )

        return base_prompt

    @staticmethod
    def build_question_prompt(
        bp: QuestionBlueprint,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        rag_context: List[Dict[str, Any]],
        existing_questions_summary: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None
    ) -> str:
        context_str = "\n".join([f"- {c.get('text', '')[:250]}" for c in rag_context[:2]]) if rag_context else "Standard NCERT curriculum content."
        # Keep only the last 4 existing questions to avoid prompt bloat
        recent_dups = existing_questions_summary[-4:] if existing_questions_summary else []
        dups_str = "\n".join([f"- {q}" for q in recent_dups]) if recent_dups else "None"

        prompt = (
            f"Generate a single question for {board.value} {class_name} {subject} ({academic_year}).\n\n"
            f"=== Blueprint Specifications ===\n"
            f"- Question Type: {bp.question_type.value}\n"
            f"- Marks Allocated: {bp.marks}\n"
            f"- Difficulty Level: {bp.difficulty.value}\n"
            f"- Chapter ID: {bp.chapter_id}\n"
            f"- Topic ID: {bp.topic_id}\n"
            f"- Bloom's Level: {bp.blooms_level}\n\n"
            f"=== Curriculum Context ===\n"
            f"{context_str}\n\n"
            f"=== Avoid Duplicating ===\n"
            f"{dups_str}\n\n"
        )

        if custom_prompt:
            prompt += f"=== Custom Instructions ===\n{custom_prompt}\n\n"

        if bp.question_type == QuestionType.MCQ:
            prompt += (
                "=== Requirements for MCQ ===\n"
                "1. Clear stem.\n"
                "2. Exactly 4 options ('A', 'B', 'C', 'D').\n"
                "3. Exactly ONE correct answer ('A', 'B', 'C', or 'D').\n"
                "4. Plausible distractors and brief explanation.\n"
            )
        else:
            prompt += (
                "=== Requirements for Question ===\n"
                "1. Unambiguous stem.\n"
                "2. Step-by-step marking scheme breakdown.\n"
                "3. Full explanation / solution.\n"
            )

        return prompt

    @staticmethod
    def build_batch_question_prompt(
        blueprints: List[QuestionBlueprint],
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        rag_context: List[Dict[str, Any]],
        custom_prompt: Optional[str] = None
    ) -> str:
        context_str = "\n".join([f"- {c.get('text', '')[:250]}" for c in rag_context[:3]]) if rag_context else "Standard NCERT curriculum content."
        spec_lines = [
            f"{i}. ID: {bp.id} | Type: {bp.question_type.value} | Marks: {bp.marks} | Difficulty: {bp.difficulty.value} | Chapter: {bp.chapter_id} | Topic: {bp.topic_id}"
            for i, bp in enumerate(blueprints, 1)
        ]
        specs_str = "\n".join(spec_lines)

        prompt = (
            f"Generate exactly {len(blueprints)} high-quality questions for {board.value} {class_name} {subject} ({academic_year}).\n\n"
            f"=== Question Specifications ===\n{specs_str}\n\n"
            f"=== Curriculum Context ===\n{context_str}\n\n"
        )
        if custom_prompt:
            prompt += f"=== Custom Instructions ===\n{custom_prompt}\n\n"

        prompt += (
            f"Return a JSON object with a 'questions' list containing exactly {len(blueprints)} questions matching each specification.\n"
            "For MCQs: exactly 4 options ('A', 'B', 'C', 'D') and correct option key.\n"
            "For Short/Long questions: step-by-step marking scheme and explanation."
        )
        return prompt


    @staticmethod
    def build_regeneration_prompt(
        bp: QuestionBlueprint,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        failed_question_stem: str,
        rejection_reason: str,
        rag_context: List[Dict[str, Any]]
    ) -> str:
        context_str = "\n".join([f"- {c.get('text', '')}" for c in rag_context]) if rag_context else "Curriculum context."
        
        return (
            f"Regenerate a NEW question for {board.value} {class_name} {subject} ({academic_year}).\n\n"
            f"The previous question was rejected during validation/quality check.\n"
            f"=== Rejected Question Stem ===\n{failed_question_stem}\n\n"
            f"=== Rejection Reason ===\n{rejection_reason}\n\n"
            f"=== Target Specifications ===\n"
            f"- Question Type: {bp.question_type.value}\n"
            f"- Marks: {bp.marks}\n"
            f"- Difficulty: {bp.difficulty.value}\n"
            f"- Chapter/Topic: {bp.chapter_id} / {bp.topic_id}\n\n"
            f"=== Curriculum Context ===\n{context_str}\n\n"
            f"DO NOT repeat the rejected question stem. Fix all flagged issues."
        )
