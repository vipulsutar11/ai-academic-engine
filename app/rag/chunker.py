import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.models.curriculum import SubjectCurriculum

logger = logging.getLogger(__name__)

class CurriculumChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(..., description="board, academic_year, class_name, subject, chapter_id, topic_id, etc.")

class CurriculumChunker:
    """Splits curriculum content into semantically coherent chunks with strict metadata tags."""

    @classmethod
    def chunk_curriculum(cls, curriculum: SubjectCurriculum) -> List[CurriculumChunk]:
        chunks: List[CurriculumChunk] = []
        chunk_counter = 1

        for chapter in curriculum.chapters:
            for topic in chapter.topics:
                # Topic text incorporating metadata context
                text = (
                    f"Board: {curriculum.board.value}\n"
                    f"Academic Year: {curriculum.academic_year}\n"
                    f"Class: {curriculum.class_name}\n"
                    f"Subject: {curriculum.subject}\n"
                    f"Unit: {chapter.unit}\n"
                    f"Chapter: {chapter.name} ({chapter.id})\n"
                    f"Topic: {topic.name} ({topic.id})\n"
                    f"Keywords: {', '.join(topic.keywords)}\n"
                    f"Learning Outcomes: {', '.join(topic.learning_outcomes)}"
                )

                metadata = {
                    "board": curriculum.board.value,
                    "academic_year": curriculum.academic_year,
                    "class_name": curriculum.class_name,
                    "subject": curriculum.subject,
                    "chapter_id": chapter.id,
                    "chapter_name": chapter.name,
                    "topic_id": topic.id,
                    "topic_name": topic.name,
                    "keywords": topic.keywords
                }

                chunk = CurriculumChunk(
                    chunk_id=f"CHK_{curriculum.board.value}_{chapter.id}_{topic.id}_{chunk_counter:04d}",
                    text=text,
                    metadata=metadata
                )
                chunks.append(chunk)
                chunk_counter += 1

        logger.info(f"Chunked curriculum into {len(chunks)} metadata-rich chunks.")
        return chunks

    @classmethod
    def chunk_text_document(
        cls,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[CurriculumChunk]:
        """Splits raw text documents into overlapping chunk windows."""
        words = text.split()
        chunks: List[CurriculumChunk] = []
        start = 0
        idx = 1

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk_meta = dict(metadata)
            chunk_id = f"TXT_CHK_{idx:04d}"

            chunks.append(CurriculumChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=chunk_meta
            ))

            start += (chunk_size - overlap)
            idx += 1

        return chunks
