import io
import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

class OCRService:
    """Multilingual OCR and PDF text extraction service supporting English, Hindi (Devanagari),
    and scanned documents via PyMuPDF (fitz), PyTesseract, and EasyOCR.
    """

    @classmethod
    def extract_text_from_bytes(cls, file_bytes: bytes, filename: str = "") -> str:
        """Extracts text from uploaded PDF or image bytes."""
        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            return cls.extract_text_from_pdf(file_bytes, filename)
        else:
            return cls.extract_text_from_image(file_bytes, filename)

    @classmethod
    def extract_text_from_pdf(cls, pdf_bytes: bytes, filename: str = "") -> str:
        """Extracts native Devanagari/Hindi and English text from PDF pages, falling back to OCR if scanned."""
        logger.info(f"Extracting text from PDF document: {filename}")
        extracted_pages = []

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text").strip()
                if len(page_text) > 20:
                    extracted_pages.append(page_text)
                else:
                    # Page may be scanned image; render page to image bytes for OCR
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    ocr_page = cls.extract_text_from_image(img_bytes, f"{filename}_page_{page_num+1}")
                    if ocr_page:
                        extracted_pages.append(ocr_page)
            
            full_pdf_text = "\n\n".join(extracted_pages).strip()
            if full_pdf_text:
                logger.info(f"Successfully extracted {len(full_pdf_text)} chars from {len(doc)} PDF pages.")
                return full_pdf_text
        except Exception as e:
            logger.warning(f"PyMuPDF PDF parsing failed: {e}. Trying raw fallback...")

        return ""

    @classmethod
    def extract_text_from_image(cls, image_bytes: bytes, filename: str = "") -> str:
        """Extracts plain Hindi/English text from image bytes using PyTesseract or EasyOCR."""
        logger.info(f"Processing Multilingual OCR for image file: {filename}")

        # 1. Open image using Pillow
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image using PIL: {e}")
            return ""

        extracted_text = ""

        # 2. Try PyTesseract first with Hindi + English support
        try:
            import pytesseract
            extracted_text = pytesseract.image_to_string(image, lang="hin+eng")
            if extracted_text and len(extracted_text.strip()) > 10:
                logger.info(f"Successfully extracted {len(extracted_text)} chars via PyTesseract (hin+eng)")
                return extracted_text.strip()
        except Exception as e:
            logger.warning(f"PyTesseract (hin+eng) unavailable: {e}. Trying EasyOCR...")

        # 3. Try EasyOCR fallback with Hindi + English support
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['hi', 'en'], gpu=False)
            img_np = np.array(image)
            results = reader.readtext(img_np, detail=0)
            extracted_text = "\n".join(results)
            if extracted_text:
                logger.info(f"Successfully extracted {len(extracted_text)} chars via EasyOCR (hi+en)")
                return extracted_text.strip()
        except Exception as e:
            logger.warning(f"EasyOCR fallback failed: {e}")

        if not extracted_text:
            logger.warning("No OCR engine available or no text found in image.")
            return f"Raw content from file {filename}."

        return extracted_text.strip()
