import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "UP"

def test_board_profiles_endpoint():
    response = client.get("/api/v1/board-profiles")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) >= 3

def test_exam_generation_endpoint():
    payload = {
        "board": "CBSE",
        "academic_year": "2024-25",
        "class_name": "Class 10",
        "subject": "Science",
        "total_marks": 10.0,
        "duration_minutes": 60,
        "title": "Class 10 Unit Test"
    }
    response = client.post("/api/v1/exam/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "exam_paper" in data["data"]

def test_question_generation_endpoint():
    payload = {
        "board": "CBSE",
        "academic_year": "2024-25",
        "class_name": "Class 10",
        "subject": "Science",
        "chapter_id": "CH_01",
        "topic_id": "CH_01_TOPIC_01",
        "question_type": "MCQ",
        "marks": 1.0,
        "difficulty": "EASY"
    }
    response = client.post("/api/v1/question/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["type"] == "MCQ"

def test_batch_exam_generation_endpoint():
    payload = {
        "requests": [
            {
                "board": "CBSE",
                "academic_year": "2024-25",
                "class_name": "Class 1",
                "subject": "Mathematics",
                "total_marks": 10.0,
                "duration_minutes": 30,
                "title": "Class 1 Math Paper"
            },
            {
                "board": "CBSE",
                "academic_year": "2024-25",
                "class_name": "Class 2",
                "subject": "Mathematics",
                "total_marks": 10.0,
                "duration_minutes": 30,
                "title": "Class 2 Math Paper"
            }
        ],
        "max_workers": 4
    }
    response = client.post("/api/v1/exam/batch-generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]["generated_papers"]) == 2
