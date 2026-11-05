import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from app.models.curriculum import BoardProfile, BoardType, QuestionType
from config import config

logger = logging.getLogger(__name__)

class BoardProfileRegistry:
    """Configuration-driven, versioned registry for academic board profiles."""
    
    _profiles: Dict[Tuple[BoardType, str], BoardProfile] = {}

    @classmethod
    def initialize_default_profiles(cls):
        """Pre-seeds standard, versioned profiles for CBSE, ICSE, and UP Board."""
        
        # CBSE 2024-25 Profile
        cbse_2024_25 = BoardProfile(
            board=BoardType.CBSE,
            academic_year="2024-25",
            supported_classes=["Class 9", "Class 10", "Class 11", "Class 12"],
            section_conventions=[
                {"name": "Section A", "type": QuestionType.MCQ, "default_marks": 1.0, "instruction": "Objective type questions / MCQs."},
                {"name": "Section B", "type": QuestionType.SHORT_ANSWER, "default_marks": 2.0, "instruction": "Very Short Answer type questions."},
                {"name": "Section C", "type": QuestionType.SHORT_ANSWER, "default_marks": 3.0, "instruction": "Short Answer type questions."},
                {"name": "Section D", "type": QuestionType.LONG_ANSWER, "default_marks": 5.0, "instruction": "Long Answer type questions."},
                {"name": "Section E", "type": QuestionType.CASE_STUDY, "default_marks": 4.0, "instruction": "Case-based / Source-based integrated questions."}
            ],
            marks_per_type_defaults={
                QuestionType.MCQ: 1.0,
                QuestionType.ASSERTION_REASON: 1.0,
                QuestionType.SHORT_ANSWER: 3.0,
                QuestionType.LONG_ANSWER: 5.0,
                QuestionType.CASE_STUDY: 4.0,
                QuestionType.NUMERICAL: 3.0
            },
            allow_internal_choice=True,
            language_mode="ENGLISH",
            pedagogical_guidelines=[
                "Follow Bloom's taxonomy with emphasis on Application and Critical Analysis.",
                "Include Assertion-Reason questions in MCQ sections.",
                "Case Study questions must test practical application of concepts."
            ]
        )

        # CBSE 2025-26 Profile
        cbse_2025_26 = cbse_2024_25.model_copy()
        cbse_2025_26.academic_year = "2025-26"
        cbse_2025_26.pedagogical_guidelines.append("Increase competency-based questions count to 50%.")

        # ICSE 2024-25 Profile
        icse_2024_25 = BoardProfile(
            board=BoardType.ICSE,
            academic_year="2024-25",
            supported_classes=["Class 9", "Class 10"],
            section_conventions=[
                {"name": "Section A", "type": QuestionType.MCQ, "default_marks": 1.0, "instruction": "Compulsory Section. Answer all questions."},
                {"name": "Section B", "type": QuestionType.SHORT_ANSWER, "default_marks": 3.0, "instruction": "Short answer questions with internal choices."},
                {"name": "Section C", "type": QuestionType.LONG_ANSWER, "default_marks": 5.0, "instruction": "Analytical and descriptive long answer questions."}
            ],
            marks_per_type_defaults={
                QuestionType.MCQ: 1.0,
                QuestionType.SHORT_ANSWER: 3.0,
                QuestionType.LONG_ANSWER: 5.0,
                QuestionType.NUMERICAL: 4.0
            },
            allow_internal_choice=True,
            language_mode="ENGLISH",
            pedagogical_guidelines=[
                "Emphasize precise definitions, SI units, and step-by-step mathematical derivations.",
                "Section A questions must cover the entire syllabus evenly."
            ]
        )

        # UP Board 2024-25 Profile
        up_board_2024_25 = BoardProfile(
            board=BoardType.UP_BOARD,
            academic_year="2024-25",
            supported_classes=["Class 9", "Class 10", "Class 11", "Class 12"],
            section_conventions=[
                {"name": "Khand K (Khanda A)", "type": QuestionType.MCQ, "default_marks": 1.0, "instruction": "Bahuvikalpiya Prashna (Multiple Choice Questions)."},
                {"name": "Khand Kh (Khanda B)", "type": QuestionType.SHORT_ANSWER, "default_marks": 2.0, "instruction": "Ati Laghu Uttariya Prashna (Very Short Answer Questions)."},
                {"name": "Khand G (Khanda C)", "type": QuestionType.SHORT_ANSWER, "default_marks": 4.0, "instruction": "Laghu Uttariya Prashna (Short Answer Questions)."},
                {"name": "Khand Gh (Khanda D)", "type": QuestionType.LONG_ANSWER, "default_marks": 6.0, "instruction": "Dirgha Uttariya Prashna (Long Answer Questions)."}
            ],
            marks_per_type_defaults={
                QuestionType.MCQ: 1.0,
                QuestionType.SHORT_ANSWER: 2.0,
                QuestionType.LONG_ANSWER: 6.0,
                QuestionType.NUMERICAL: 4.0
            },
            allow_internal_choice=True,
            language_mode="BILINGUAL",  # Supports Hindi + English context
            pedagogical_guidelines=[
                "Provide technical terms in both Hindi and English where helpful.",
                "Focus on core textbook concepts and NCERT/SCERT aligned problems."
            ]
        )

        cls.register_profile(cbse_2024_25)
        cls.register_profile(cbse_2025_26)
        cls.register_profile(icse_2024_25)
        cls.register_profile(up_board_2024_25)

    @classmethod
    def register_profile(cls, profile: BoardProfile):
        key = (profile.board, profile.academic_year)
        cls._profiles[key] = profile
        logger.info(f"Registered board profile for {profile.board} - Year {profile.academic_year}")

    @classmethod
    def get_profile(cls, board: BoardType, academic_year: str) -> BoardProfile:
        if not cls._profiles:
            cls.initialize_default_profiles()
        
        key = (board, academic_year)
        if key in cls._profiles:
            return cls._profiles[key]
        
        # Fallback to matching board with latest available year
        for (b, yr), profile in cls._profiles.items():
            if b == board:
                logger.warning(f"Exact match for profile {board} year {academic_year} not found. Using profile for year {yr}.")
                return profile
        
        raise ValueError(f"No profile found for board: {board} and year: {academic_year}")

    @classmethod
    def load_custom_profile_from_file(cls, file_path: Path) -> BoardProfile:
        """Loads custom BoardProfile JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = BoardProfile.model_validate(data)
        cls.register_profile(profile)
        return profile

# Helper function
def get_board_profile(board: BoardType, academic_year: str = "2024-25") -> BoardProfile:
    return BoardProfileRegistry.get_profile(board, academic_year)
