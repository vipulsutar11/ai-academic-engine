import os
import sys
import zipfile
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pymupdf  # PyMuPDF for PDF parsing
from app.curriculum.loader import CurriculumLoader
from app.rag.chunker import CurriculumChunker, CurriculumChunk
from app.rag.retriever import RAGRetriever
from app.models.curriculum import BoardType
from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_drive_zips")

DOWNLOAD_DIR = config.DATA_DIR / "downloads"
EXTRACT_DIR = config.DATA_DIR / "extracted"

def extract_all_zips(download_dir: Path, extract_dir: Path) -> List[Path]:
    """Extracts all ZIP files found in download_dir to extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_files = list(download_dir.glob("*.zip"))
    
    extracted_dirs = []
    logger.info(f"Found {len(zip_files)} zip file(s) in {download_dir}")
    
    for zip_path in zip_files:
        target_subfolder = extract_dir / zip_path.stem
        target_subfolder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Unzipping {zip_path.name} -> {target_subfolder}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_subfolder)
            extracted_dirs.append(target_subfolder)
        except Exception as e:
            logger.error(f"Failed to extract {zip_path.name}: {e}")
            
    return extracted_dirs

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts plain text from PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(pdf_path)
        text_pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_pages.append(page.get_text())
        return "\n".join(text_pages)
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path.name}: {e}")
        return ""

def process_extracted_files(extract_dir: Path) -> List[Dict[str, Any]]:
    """Recursively processes PDF, JSON, and TXT files and generates RAG chunks."""
    all_chunks: List[Dict[str, Any]] = []
    
    # 1. Process JSON files
    json_files = list(extract_dir.rglob("*.json"))
    logger.info(f"Processing {len(json_files)} JSON file(s)...")
    for jf in json_files:
        try:
            curriculum = CurriculumLoader.load_from_json_file(jf)
            chunks = CurriculumChunker.chunk_curriculum(curriculum)
            for c in chunks:
                all_chunks.append({"text": c.text, "metadata": c.metadata})
        except Exception:
            # Fallback to plain JSON string chunking if schema isn't SubjectCurriculum
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    content = f.read()
                raw_chunks = CurriculumChunker.chunk_text_document(
                    text=content,
                    metadata={"source_file": jf.name, "file_type": "json"},
                    chunk_size=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP
                )
                for c in raw_chunks:
                    all_chunks.append({"text": c.text, "metadata": c.metadata})
            except Exception as ex:
                logger.error(f"Error processing JSON file {jf}: {ex}")

    # 2. Process PDF files
    pdf_files = list(extract_dir.rglob("*.pdf"))
    logger.info(f"Processing {len(pdf_files)} PDF file(s)...")
    for pdf_path in pdf_files:
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            continue
            
        metadata = {
            "source_file": pdf_path.name,
            "source_dir": pdf_path.parent.name,
            "board": BoardType.CBSE.value,
            "file_type": "pdf"
        }
        
        raw_chunks = CurriculumChunker.chunk_text_document(
            text=raw_text,
            metadata=metadata,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        for c in raw_chunks:
            all_chunks.append({"text": c.text, "metadata": c.metadata})

    # 3. Process TXT files
    txt_files = list(extract_dir.rglob("*.txt"))
    logger.info(f"Processing {len(txt_files)} TXT file(s)...")
    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content.strip():
                continue
            metadata = {
                "source_file": txt_path.name,
                "file_type": "txt"
            }
            raw_chunks = CurriculumChunker.chunk_text_document(
                text=content,
                metadata=metadata,
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )
            for c in raw_chunks:
                all_chunks.append({"text": c.text, "metadata": c.metadata})
        except Exception as e:
            logger.error(f"Failed to process text file {txt_path}: {e}")

    return all_chunks

def main():
    logger.info("=== Starting Drive Zip Ingestion Pipeline ===")
    
    # Step 1: Extract Zips
    extract_all_zips(DOWNLOAD_DIR, EXTRACT_DIR)
    
    # Step 2: Process files and generate chunks
    chunks = process_extracted_files(EXTRACT_DIR)
    logger.info(f"Total chunks generated across all documents: {len(chunks)}")
    
    if not chunks:
        logger.warning("No chunks generated. Check download and extract folders.")
        return
        
    # Step 3: Embed & inject into Vector DB
    retriever = RAGRetriever()
    logger.info(f"Embedding and indexing {len(chunks)} chunks into Vector DB...")
    retriever.index_curriculum_chunks(chunks)
    
    # Step 4: Persist Vector DB index to disk
    store_dir = config.VECTOR_STORE_DIR
    retriever.vector_store.save(store_dir)
    logger.info(f"=== Successfully saved vector index to {store_dir} ===")

if __name__ == "__main__":
    main()
