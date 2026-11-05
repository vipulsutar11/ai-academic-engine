import logging
import numpy as np
from typing import List, Union
from config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Embedding Generator service using sentence-transformers with fallback for offline execution."""

    _model = None
    _dim: int = 384  # Default dimension for all-MiniLM-L6-v2

    _cache = {}

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._load_model()

    def _load_model(self):
        if EmbeddingService._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                EmbeddingService._model = SentenceTransformer(self.model_name)
                if hasattr(EmbeddingService._model, "get_embedding_dimension"):
                    EmbeddingService._dim = EmbeddingService._model.get_embedding_dimension()
                else:
                    EmbeddingService._dim = EmbeddingService._model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Could not load sentence-transformers model '{self.model_name}': {e}. Using lightweight fallback vector generator.")
                EmbeddingService._model = "FALLBACK"
                EmbeddingService._dim = 384

    @property
    def dimension(self) -> int:
        return EmbeddingService._dim

    def _encode_uncached(self, texts: List[str]) -> np.ndarray:
        if EmbeddingService._model != "FALLBACK" and EmbeddingService._model is not None:
            try:
                embeddings = EmbeddingService._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"Embedding encoding failed: {e}. Falling back to deterministic pseudo-embeddings.")

        # Fallback deterministic pseudo-embeddings using hash algorithm
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for i, char in enumerate(text.encode("utf-8")):
                vec[i % self.dimension] += ord(char)
            # Normalize vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        unique_uncached = []
        text_to_indices = {}
        result_vecs = [None] * len(texts)

        for idx, text in enumerate(texts):
            if text in EmbeddingService._cache:
                result_vecs[idx] = EmbeddingService._cache[text]
            else:
                if text not in text_to_indices:
                    text_to_indices[text] = []
                    unique_uncached.append(text)
                text_to_indices[text].append(idx)

        if unique_uncached:
            new_embeddings = self._encode_uncached(unique_uncached)
            for text, emb in zip(unique_uncached, new_embeddings):
                EmbeddingService._cache[text] = emb
                for idx in text_to_indices[text]:
                    result_vecs[idx] = emb

        return np.array(result_vecs, dtype=np.float32)
