import pytest
from pathlib import Path
from app.models.curriculum import BoardType, SubjectCurriculum
from app.curriculum.board_profiles import BoardProfileRegistry, get_board_profile
from app.curriculum.loader import CurriculumLoader

def test_board_profile_registry():
    cbse_profile = get_board_profile(BoardType.CBSE, "2024-25")
    assert cbse_profile.board == BoardType.CBSE
    assert cbse_profile.academic_year == "2024-25"
    assert len(cbse_profile.section_conventions) >= 4

    icse_profile = get_board_profile(BoardType.ICSE, "2024-25")
    assert icse_profile.board == BoardType.ICSE

    up_profile = get_board_profile(BoardType.UP_BOARD, "2024-25")
    assert up_profile.board == BoardType.UP_BOARD
    assert up_profile.language_mode == "BILINGUAL"

def test_load_curriculum_from_json():
    json_path = Path(__file__).parent.parent / "data" / "curriculum" / "cbse" / "class10_science_2024_25.json"
    if json_path.exists():
        curriculum = CurriculumLoader.load_from_json_file(json_path)
        assert curriculum.board == BoardType.CBSE
        assert curriculum.class_name == "Class 10"
        assert len(curriculum.chapters) > 0

def test_parse_raw_text_curriculum():
    raw_text = """
    Chapter 1: Forces and Motion
    - Newton's First Law of Motion
    - Newton's Second Law of Motion
    Chapter 2: Energy and Work
    - Kinetic and Potential Energy
    """
    curriculum = CurriculumLoader.parse_raw_text(
        raw_text=raw_text,
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Physics"
    )
    assert len(curriculum.chapters) == 2
    assert curriculum.chapters[0].name.startswith("Chapter 1")
    assert len(curriculum.chapters[0].topics) == 2
