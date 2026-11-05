import json
import logging
from pathlib import Path
from typing import Union, Dict, Any, List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Document loader supporting PDF (using PyMuPDF), JSON, and text syllabus files."""

    @staticmethod
    def load_pdf(file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")

        logger.info(f"Loading PDF document using PyMuPDF: {path}")
        doc = fitz.open(path)
        pages_text: List[str] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            if text.strip():
                pages_text.append(text)
        doc.close()
        return "\n\n".join(pages_text)

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_text(file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
