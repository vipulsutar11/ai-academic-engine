import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from app.api.main import app
from app.ocr.service import OCRService

client = TestClient(app)

def create_sample_image_bytes():
    """Generates an in-memory sample image with text for testing."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 40), "Chemical Reactions and Equations Class 10 Science", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_ocr_service_extraction():
    image_bytes = create_sample_image_bytes()
    text = OCRService.extract_text_from_bytes(image_bytes, filename="test_sample.png")
    assert isinstance(text, str)
    assert len(text) > 0

def test_generate_exam_from_image_endpoint():
    image_bytes = create_sample_image_bytes()
    files = {
        "file": ("test_sample.png", image_bytes, "image/png")
    }
    data = {
        "prompt": "Create a 20-mark quiz on Chemical Reactions",
        "board": "CBSE",
        "class_name": "Class 10",
        "subject": "Science",
        "total_marks": 20,
        "language_mode": "HINDI"
    }
    response = client.post("/api/v1/exam/generate-from-image", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert "ocr_extracted_text" in res_json["data"]
    assert "exam_paper" in res_json["data"]
