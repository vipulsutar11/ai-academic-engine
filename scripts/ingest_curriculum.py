import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from app.curriculum.loader import CurriculumLoader
from app.rag.chunker import CurriculumChunker
from app.rag.retriever import RAGRetriever
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting curriculum ingestion script...")
    retriever = RAGRetriever()

    curriculum_dir = config.CURRICULUM_DIR
    json_files = list(curriculum_dir.rglob("*.json"))

    if not json_files:
        logger.warning(f"No curriculum JSON files found in {curriculum_dir}")
        return

    total_chunks = 0
    for file_path in json_files:
        logger.info(f"Processing curriculum file: {file_path}")
        curriculum = CurriculumLoader.load_from_json_file(file_path)
        chunks = CurriculumChunker.chunk_curriculum(curriculum)
        raw_chunks = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        retriever.index_curriculum_chunks(raw_chunks)
        total_chunks += len(raw_chunks)

    # Persist vector store to disk
    store_dir = config.VECTOR_STORE_DIR
    retriever.vector_store.save(store_dir)
    logger.info(f"Ingestion complete. Indexed and saved {total_chunks} chunks to {store_dir}")

if __name__ == "__main__":
    main()
