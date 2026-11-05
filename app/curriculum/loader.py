import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

from app.models.curriculum import SubjectCurriculum, Chapter, Topic, BoardType

logger = logging.getLogger(__name__)

class CurriculumLoader:
    """Loads and parses curriculum documents (JSON, PDF, TXT) into SubjectCurriculum domain models."""

    @staticmethod
    def load_from_json_file(file_path: Path) -> SubjectCurriculum:
        """Loads SubjectCurriculum from a structured JSON file."""
        logger.info(f"Loading curriculum from JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SubjectCurriculum.model_validate(data)

    @staticmethod
    def load_from_pdf(
        file_path: Path,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str
    ) -> SubjectCurriculum:
        """Extracts text from PDF using PyMuPDF and parses chapters/topics."""
        logger.info(f"Extracting curriculum from PDF document: {file_path}")
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n"

        return CurriculumLoader.parse_raw_text(
            raw_text=full_text,
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject
        )

    @staticmethod
    def parse_raw_text(
        raw_text: str,
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str
    ) -> SubjectCurriculum:
        """Parses structured text into a SubjectCurriculum model."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        chapters: List[Chapter] = []
        current_chapter: Optional[Chapter] = None
        chapter_idx = 1
        topic_idx = 1

        for line in lines:
            if line.lower().startswith("chapter") or line.lower().startswith("unit") or line.startswith("मझ्य"):
                if current_chapter:
                    chapters.append(current_chapter)
                ch_id = f"CH_{chapter_idx:02d}"
                current_chapter = Chapter(
                    id=ch_id,
                    name=line,
                    unit=f"Unit {chapter_idx}",
                    topics=[]
                )
                chapter_idx += 1
                topic_idx = 1
            elif current_chapter and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                clean_topic = line.lstrip("-*0123456789. ").strip()
                if clean_topic:
                    t_id = f"{current_chapter.id}_TOPIC_{topic_idx:02d}"
                    topic = Topic(
                        id=t_id,
                        name=clean_topic,
                        weightage=1.0,
                        keywords=[word.lower() for word in clean_topic.split() if len(word) > 3]
                    )
                    current_chapter.topics.append(topic)
                    topic_idx += 1

        if current_chapter:
            chapters.append(current_chapter)

        # Fallback if no explicit chapter headers found
        if not chapters:
            fallback_ch = Chapter(
                id="CH_01",
                name="General Syllabus",
                unit="Unit 1",
                topics=[
                    Topic(id="CH_01_TOPIC_01", name="Core Curriculum Concepts", weightage=1.0, keywords=["general", "core"])
                ]
            )
            chapters.append(fallback_ch)

        return SubjectCurriculum(
            board=board,
            academic_year=academic_year,
            class_name=class_name,
            subject=subject,
            chapters=chapters
        )

    @staticmethod
    def find_and_load_curriculum(
        board: BoardType,
        academic_year: str,
        class_name: str,
        subject: str,
        curriculum_dir: Optional[Path] = None
    ) -> Optional[SubjectCurriculum]:
        """Automatically searches for and loads matching curriculum JSON from disk."""
        from config import config
        root_dir = curriculum_dir or (config.DATA_DIR / "curriculum")
        board_val = board.value.lower() if hasattr(board, "value") else str(board).lower()

        board_dir = root_dir / board_val
        search_dirs = [board_dir, root_dir] if board_dir.exists() else [root_dir]

        clean_subj = subject.lower().replace(" ", "").replace("-", "_")
        clean_cls = "".join(filter(str.isalnum, class_name.lower()))

        for s_dir in search_dirs:
            for jf in s_dir.rglob("*.json"):
                name_clean = jf.stem.lower().replace("-", "_")
                if clean_subj in name_clean:
                    if clean_cls in name_clean or clean_cls.replace("class", "") in name_clean:
                        try:
                            curr = CurriculumLoader.load_from_json_file(jf)
                            logger.info(f"Loaded curriculum '{curr.subject}' for {curr.board} {curr.class_name} from {jf.name}")
                            return curr
                        except Exception as e:
                            logger.warning(f"Failed to load candidate curriculum file {jf}: {e}")
        return None
