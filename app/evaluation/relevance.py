import logging
import numpy as np
from typing import List, Dict, Any, Optional
from app.rag.embeddings import EmbeddingService
from app.models.evaluation import RelevanceResult

logger = logging.getLogger(__name__)

class RelevanceEvaluator:
    """Evaluates question semantic relevance against RAG retrieved curriculum context."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()

    def evaluate_relevance(
        self,
        question_text: str,
        rag_context: List[Dict[str, Any]],
        threshold: float = 0.40
    ) -> RelevanceResult:
        if not rag_context:
            return RelevanceResult(is_relevant=True, similarity_score=1.0, matched_chunks=[])

        context_texts = [c.get("text", "") for c in rag_context]
        q_embed = self.embedding_service.encode(question_text).flatten()
        ctx_embeds = self.embedding_service.encode(context_texts)

        q_norm = np.linalg.norm(q_embed)
        if q_norm > 0:
            q_embed = q_embed / q_norm

        norms = np.linalg.norm(ctx_embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_ctx = ctx_embeds / norms

        sims = np.dot(norm_ctx, q_embed)
        max_sim = float(np.max(sims))

        is_rel = max_sim >= threshold
        matched = [context_texts[i] for i, s in enumerate(sims) if s >= threshold]

        return RelevanceResult(
            is_relevant=is_rel,
            similarity_score=max_sim,
            matched_chunks=matched
        )
