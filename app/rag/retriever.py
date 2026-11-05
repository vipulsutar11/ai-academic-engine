import logging
from typing import List, Dict, Any, Optional
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import BaseVectorStore, VectorStoreFactory
from app.models.curriculum import BoardType

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Curriculum-aware retriever filtering by Board, Academic Year, Class, Subject, Chapter & Topic metadata."""

    def __init__(self, vector_store: Optional[BaseVectorStore] = None, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreFactory.create()
        self.vector_store.load()

    def index_curriculum_chunks(self, chunks: List[Dict[str, Any]], persist: bool = False):
        """Indexes curriculum chunks into the vector store and optionally persists to disk."""
        if not chunks:
            return
        texts = [c.get("text", "") for c in chunks]
        embeddings = self.embedding_service.encode(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        if persist:
            self.vector_store.save()
        logger.info(f"Indexed {len(chunks)} curriculum chunks into RAG store (persist={persist}).")

    def retrieve(
        self,
        query: str,
        board: Optional[BoardType] = None,
        academic_year: Optional[str] = None,
        class_name: Optional[str] = None,
        subject: Optional[str] = None,
        chapter_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant context chunks applying strict metadata filters."""
        query_embedding = self.embedding_service.encode(query)

        filters: Dict[str, Any] = {}
        if board:
            filters["board"] = board.value if isinstance(board, BoardType) else board
        if academic_year:
            filters["academic_year"] = academic_year
        if class_name:
            filters["class_name"] = class_name
        if subject:
            filters["subject"] = subject
        if chapter_id:
            filters["chapter_id"] = chapter_id
        if topic_id:
            filters["topic_id"] = topic_id

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        retrieved_chunks = []
        for chunk, score in results:
            chunk_copy = dict(chunk)
            chunk_copy["similarity_score"] = float(score)
            retrieved_chunks.append(chunk_copy)

        logger.info(f"RAG query retrieved {len(retrieved_chunks)} relevant chunks with filters: {filters}")
        return retrieved_chunks
