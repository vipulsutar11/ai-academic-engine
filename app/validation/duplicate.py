import logging
import numpy as np
from typing import List, Tuple, Optional
from app.rag.embeddings import EmbeddingService
from app.models.question import Question
from config import config

logger = logging.getLogger(__name__)

class SemanticDuplicateDetector:
    """Semantic duplicate detection using embedding similarity."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.threshold = threshold if threshold is not None else config.DUPLICATE_SIMILARITY_THRESHOLD

    def is_duplicate(
        self,
        candidate_question: Question,
        existing_questions: List[Question]
    ) -> Tuple[bool, float, Optional[Question]]:
        """Checks if candidate_question is a semantic duplicate of any existing question."""
        if not existing_questions:
            return False, 0.0, None

        # Build text representation including options if MCQ
        candidate_text = self._build_question_text(candidate_question)
        existing_texts = [self._build_question_text(q) for q in existing_questions]

        # Embed texts
        candidate_embed = self.embedding_service.encode(candidate_text).flatten()
        existing_embeds = self.embedding_service.encode(existing_texts)

        # Normalize
        c_norm = np.linalg.norm(candidate_embed)
        if c_norm > 0:
            candidate_embed = candidate_embed / c_norm

        norms = np.linalg.norm(existing_embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_existing = existing_embeds / norms

        # Cosine similarities
        similarities = np.dot(norm_existing, candidate_embed)
        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])

        is_dup = max_similarity >= self.threshold
        matched_q = existing_questions[max_idx] if is_dup else None

        if is_dup:
            logger.warning(
                f"Duplicate detected: Question stem '{candidate_question.stem[:40]}...' is {max_similarity:.3f} similar to existing question '{matched_q.id}' (threshold: {self.threshold})."
            )

        return is_dup, max_similarity, matched_q

    def find_duplicates_in_paper(
        self,
        questions: List[Question]
    ) -> List[Tuple[Question, Question, float]]:
        """Vectorized batch duplicate detection for all questions in an exam paper.
        Encodes all questions in a single matrix embedding call (sub-second) instead of
        quadratic incremental loop calls.
        Returns list of (duplicate_question, original_question, similarity_score).
        """
        if len(questions) <= 1:
            return []

        texts = [self._build_question_text(q) for q in questions]
        embeds = self.embedding_service.encode(texts)

        # Normalize embeddings row-wise
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeds = embeds / norms

        # Compute pairwise cosine similarity matrix in 1 BLAS call
        sim_matrix = np.dot(norm_embeds, norm_embeds.T)

        duplicates = []
        duplicate_indices = set()

        for i in range(len(questions)):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, len(questions)):
                if j in duplicate_indices:
                    continue
                sim = float(sim_matrix[i, j])
                if sim >= self.threshold:
                    duplicate_indices.add(j)
                    duplicates.append((questions[j], questions[i], sim))
                    logger.warning(
                        f"Duplicate detected: Question '{questions[j].id}' is {sim:.3f} similar to '{questions[i].id}' (threshold: {self.threshold})."
                    )

        return duplicates

    @staticmethod
    def _build_question_text(q: Question) -> str:
        text = f"{q.stem}"
        if q.options:
            opts_str = " ".join([f"({opt.key}) {opt.text}" for opt in q.options])
            text += f" {opts_str}"
        return text
