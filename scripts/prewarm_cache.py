import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from app.models.curriculum import BoardType, QuestionType, DifficultyLevel
from app.models.blueprint import QuestionBlueprint
from app.curriculum.loader import CurriculumLoader
from app.generators.question import QuestionGenerator
from app.ai.provider import BaseLLMProvider
from app.ai.mistral import MistralProvider
from app.ai.gemini import GeminiProvider
from app.ai.mock_provider import MockLLMProvider
from app.ai.fallback import TwoTierFallbackProvider
from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prewarm_cache")

def get_provider() -> BaseLLMProvider:
    def _is_real_key(key: str) -> bool:
        if not key:
            return False
        k = key.strip().lower()
        return not (k.startswith("mock") or k.startswith("your_") or "here" in k or "<" in k)

    primary = MistralProvider() if _is_real_key(config.MISTRAL_API_KEY) else MockLLMProvider()
    fallback = GeminiProvider() if _is_real_key(config.GEMINI_API_KEY) else MockLLMProvider()

    return TwoTierFallbackProvider(
        primary_provider=primary,
        secondary_provider=fallback,
        tertiary_provider=MockLLMProvider()
    )

def prewarm_subject_curriculum(board: BoardType, class_name: str, subject: str, limit_topics: int = 5):
    """Pre-generates and caches verified questions for high-frequency curriculum topics."""
    logger.info(f"Pre-warming cache for {board.value} {class_name} {subject}...")
    curr = CurriculumLoader.find_and_load_curriculum(
        board=board,
        academic_year="2024-25",
        class_name=class_name,
        subject=subject
    )
    if not curr:
        logger.warning(f"No curriculum found for {board} {class_name} {subject}. Skipping.")
        return

    provider = get_provider()
    gen = QuestionGenerator(provider=provider)

    count = 0
    for ch in curr.chapters:
        for topic in ch.topics[:limit_topics]:
            # Warm up 1 MCQ and 1 Short Answer for each topic
            for q_type, default_marks, diff in [
                (QuestionType.MCQ, 1.0, DifficultyLevel.EASY),
                (QuestionType.MCQ, 1.0, DifficultyLevel.MEDIUM),
                (QuestionType.SHORT_ANSWER, 2.0, DifficultyLevel.MEDIUM)
            ]:
                bp = QuestionBlueprint(
                    id=f"PREWARM_{ch.id}_{topic.id}_{q_type.value}",
                    question_type=q_type,
                    marks=default_marks,
                    difficulty=diff,
                    chapter_id=ch.id,
                    topic_id=topic.id,
                    blooms_level="Application" if diff == DifficultyLevel.MEDIUM else "Knowledge"
                )
                try:
                    q = gen.generate_question(
                        bp=bp,
                        board=board,
                        academic_year="2024-25",
                        class_name=class_name,
                        subject=subject
                    )
                    count += 1
                    logger.info(f"Cached {q.type.value} for {ch.name} - {topic.name} ({count} total)")
                except Exception as e:
                    logger.warning(f"Failed to generate pre-warm question: {e}")

    gen.question_cache.save_to_disk()
    logger.info(f"Pre-warming complete! Stored {count} verified questions to QuestionBankCache.")

if __name__ == "__main__":
    prewarm_subject_curriculum(BoardType.CBSE, "Class 10", "History", limit_topics=2)
