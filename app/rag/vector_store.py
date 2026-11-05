import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

class BaseVectorStore(ABC):
    """Replaceable Vector Store interface for curriculum chunk embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add chunks with their corresponding embeddings and metadata."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search top-k similar chunks filtered by metadata fields."""
        pass

    @abstractmethod
    def save(self, directory: Path):
        """Persist index to disk."""
        pass

    @abstractmethod
    def load(self, directory: Path):
        """Load index from disk."""
        pass


class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store using Cosine Similarity for fast execution & testing."""

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) == 0:
            return
        
        self.chunks.extend(chunks)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Ensure 1D query
        q_vec = query_embedding.flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Filter indices by metadata
        candidate_indices = []
        for i, chunk in enumerate(self.chunks):
            meta = chunk.get("metadata", {})
            match = True
            if filters:
                for k, v in filters.items():
                    if v is not None and meta.get(k) != v:
                        match = False
                        break
            if match:
                candidate_indices.append(i)

        if not candidate_indices:
            return []

        # Calculate cosine similarity with candidate embeddings
        cand_embeds = self.embeddings[candidate_indices]
        norms = np.linalg.norm(cand_embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeds = cand_embeds / norms

        sims = np.dot(normalized_embeds, q_vec)

        # Sort candidate indices by similarity descending
        sorted_pairs = sorted(zip(candidate_indices, sims), key=lambda x: x[1], reverse=True)
        top_pairs = sorted_pairs[:top_k]

        return [(self.chunks[idx], float(sim)) for idx, sim in top_pairs]

    def save(self, directory: Optional[Path] = None):
        target_dir = Path(directory) if directory is not None else config.VECTOR_STORE_DIR
        os.makedirs(target_dir, exist_ok=True)
        with open(target_dir / "memory_store_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        if self.embeddings is not None:
            np.save(target_dir / "memory_store_embeds.npy", self.embeddings)

    def load(self, directory: Optional[Path] = None):
        target_dir = Path(directory) if directory is not None else config.VECTOR_STORE_DIR
        chunks_file = target_dir / "memory_store_chunks.json"
        embeds_file = target_dir / "memory_store_embeds.npy"
        if chunks_file.exists() and embeds_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(embeds_file)


class FAISSVectorStore(BaseVectorStore):
    """Production FAISS Vector Store implementation with metadata filtering."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks: List[Dict[str, Any]] = []
        self._index = None
        self._init_faiss()

    def _init_faiss(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized cosine similarity
        except ImportError:
            logger.warning("FAISS package not found. Falling back to MemoryVectorStore.")
            self._fallback_store = MemoryVectorStore()

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if hasattr(self, "_fallback_store"):
            return self._fallback_store.add_chunks(chunks, embeddings)

        if len(chunks) == 0:
            return

        # Normalize for Inner Product (Cosine Similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embeds = (embeddings / norms).astype(np.float32)

        self.chunks.extend(chunks)
        self._index.add(norm_embeds)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        if hasattr(self, "_fallback_store"):
            return self._fallback_store.search(query_embedding, top_k, filters)

        if self._index is None or self._index.ntotal == 0:
            return []

        q_vec = query_embedding.flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        q_vec = np.expand_dims(q_vec, axis=0).astype(np.float32)

        # Retrieve top 50 candidates for metadata filtering
        k_search = min(max(top_k * 5, 50), self._index.ntotal)
        scores, indices = self._index.search(q_vec, k_search)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            meta = chunk.get("metadata", {})

            # Filter metadata
            match = True
            if filters:
                for k, v in filters.items():
                    if v is not None and meta.get(k) != v:
                        match = False
                        break
            if match:
                results.append((chunk, float(score)))
                if len(results) >= top_k:
                    break

        return results

    def save(self, directory: Optional[Path] = None):
        if hasattr(self, "_fallback_store"):
            return self._fallback_store.save(directory)

        target_dir = Path(directory) if directory is not None else config.VECTOR_STORE_DIR
        os.makedirs(target_dir, exist_ok=True)
        import faiss
        faiss.write_index(self._index, str(target_dir / "faiss.index"))
        with open(target_dir / "faiss_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def load(self, directory: Optional[Path] = None):
        if hasattr(self, "_fallback_store"):
            return self._fallback_store.load(directory)

        target_dir = Path(directory) if directory is not None else config.VECTOR_STORE_DIR
        index_file = target_dir / "faiss.index"
        chunks_file = target_dir / "faiss_chunks.json"
        if index_file.exists() and chunks_file.exists():
            import faiss
            self._index = faiss.read_index(str(index_file))
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks from {target_dir}")


class VectorStoreFactory:
    @staticmethod
    def create(store_type: str = None, dimension: int = 384) -> BaseVectorStore:
        t = (store_type or config.VECTOR_STORE_TYPE).lower()
        if t == "faiss":
            return FAISSVectorStore(dimension=dimension)
        return MemoryVectorStore()
