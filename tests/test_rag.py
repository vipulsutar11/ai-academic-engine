import pytest
import numpy as np
from app.models.curriculum import BoardType, SubjectCurriculum, Chapter, Topic
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import MemoryVectorStore, FAISSVectorStore, VectorStoreFactory
from app.rag.chunker import CurriculumChunker
from app.rag.retriever import RAGRetriever

def test_embedding_service():
    service = EmbeddingService()
    vecs = service.encode(["Ohm's Law states V = IR", "Magnetic effect of electric current"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 2
    assert vecs.shape[1] == service.dimension

def test_vector_store_memory():
    store = MemoryVectorStore()
    chunk1 = {"text": "Ohm's Law V = IR", "metadata": {"board": "CBSE", "subject": "Science"}}
    chunk2 = {"text": "Photosynthesis in plants", "metadata": {"board": "ICSE", "subject": "Biology"}}

    embedder = EmbeddingService()
    embeds = embedder.encode([chunk1["text"], chunk2["text"]])
    store.add_chunks([chunk1, chunk2], embeds)

    q_embed = embedder.encode("electric voltage and resistance")
    results = store.search(q_embed, top_k=1, filters={"board": "CBSE"})
    
    assert len(results) == 1
    assert results[0][0]["metadata"]["board"] == "CBSE"

def test_rag_retriever():
    curriculum = SubjectCurriculum(
        board=BoardType.CBSE,
        academic_year="2024-25",
        class_name="Class 10",
        subject="Science",
        chapters=[
            Chapter(
                id="CH_01",
                name="Electricity",
                unit="Unit 1",
                topics=[Topic(id="CH_01_TOPIC_01", name="Ohm's Law", weightage=1.0, keywords=["ohm", "current"])]
            )
        ]
    )

    chunks = CurriculumChunker.chunk_curriculum(curriculum)
    retriever = RAGRetriever()
    retriever.index_curriculum_chunks([{"text": c.text, "metadata": c.metadata} for c in chunks])

    results = retriever.retrieve(
        query="Ohm's law formula",
        board=BoardType.CBSE,
        subject="Science",
        top_k=1
    )
    assert len(results) == 1
    assert results[0]["metadata"]["board"] == "CBSE"
