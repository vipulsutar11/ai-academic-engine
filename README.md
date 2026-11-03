# AI Academic Content & Examination Agent

Production-ready AI/ML engine for automated, curriculum-aware generation of academic examination papers, MCQs, assignments, individual questions, answer keys, and marking schemes for **CBSE**, **ICSE**, and **UP Board** across versioned academic years.

---

## 🏛️ Architecture Overview

```text
Input Request
    ↓
Board Profile Registry (CBSE / ICSE / UP Board, Academic Year)
    ↓
Blueprint Engine (Deterministic Marks, Section & Question Breakdown)
    ↓
RAG Retrieval (PyMuPDF Chunking & Sentence-Transformers Embeddings)
    ↓
LLM Provider (OpenAI-compatible / Mock Provider Abstraction)
    ↓
Structured Output Parsing (Strict Pydantic JSON Schemas)
    ↓
Deterministic Validation (Schema, Marks Sums, Difficulty & Topic Distribution)
    ↓
Semantic Duplicate Detection (Embedding Cosine Similarity Thresholding)
    ↓
AI Quality & Relevance Evaluation (Pedagogical Validity & Alignment Scoring)
    ↓
Targeted Question Regeneration (Iterative Single-Question Healing)
    ↓
Validated Production-Ready Academic Content
```

---

## 📁 Project Structure

```text
ai-academic-engine/
├── app/
│   ├── api/                   # FastAPI routes & web server main
│   │   ├── main.py
│   │   └── routes.py
│   ├── generators/            # Academic content generators
│   │   ├── exam.py
│   │   ├── mcq.py
│   │   ├── assignment.py
│   │   ├── question.py
│   │   └── answer.py
│   ├── rag/                   # RAG Pipeline (Loader, Chunker, Embeddings, Store, Retriever)
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── blueprint/             # Deterministic Blueprint Engine & Validator
│   │   ├── engine.py
│   │   └── validator.py
│   ├── curriculum/            # Configuration-driven Board Profiles & Syllabus Loaders
│   │   ├── loader.py
│   │   └── board_profiles.py
│   ├── ai/                    # LLM Provider Abstraction & Prompts
│   │   ├── provider.py
│   │   ├── mistral.py
│   │   ├── gemini.py
│   │   ├── fallback.py
│   │   ├── mock_provider.py
│   │   ├── prompts.py
│   │   └── retry.py
│   ├── validation/            # Deterministic Validators & Duplicate Detection
│   │   ├── schema.py
│   │   ├── marks.py
│   │   ├── difficulty.py
│   │   ├── topic.py
│   │   ├── duplicate.py
│   │   └── paper.py
│   ├── evaluation/            # AI Quality & Relevance Evaluators
│   │   ├── quality.py
│   │   └── relevance.py
│   ├── models/                # Pydantic Domain Schemas & API DTOs
│   │   ├── curriculum.py
│   │   ├── blueprint.py
│   │   ├── question.py
│   │   ├── exam.py
│   │   ├── assignment.py
│   │   ├── evaluation.py
│   │   └── api_models.py
│   └── pipeline.py            # Master Production Pipeline Orchestrator
│
├── data/                      # Sample curriculum JSONs & pre-seeded data
│   ├── curriculum/
│   │   ├── cbse/
│   │   ├── icse/
│   │   └── up_board/
│   └── blueprints/
│
├── scripts/                   # Ingestion CLI tools
│   └── ingest_curriculum.py
├── tests/                     # Pytest suite
├── requirements.txt
├── .env.example
├── config.py
└── README.md
```

---

## ⚡ Key Features & Requirements Satisfied

1. **Board Profile Registry**: Configuration-driven and versioned profiles for CBSE, ICSE, and UP Board (including bilingual Hindi/English for UP Board).
2. **Deterministic Blueprint Engine**: Blueprint Engine dictates total marks, question types, section counts, marks distribution, and difficulty levels. LLM never freely decides these.
3. **RAG Pipeline**: Curriculum PDF parsing via PyMuPDF (`fitz`), chunking with rich metadata tags (`board`, `academic_year`, `class_name`, `subject`, `chapter_id`, `topic_id`), sentence-transformers embeddings, and replaceable vector store (FAISS & Memory).
4. **Targeted Question Regeneration**: Replaces only failing or duplicate questions without regenerating the entire paper.
5. **Deterministic & AI Quality Validation**: Validates schema compliance, total mark balance, section question counts, difficulty ratios, topic coverage, and semantic duplicates.
6. **Primary Mistral & Gemini Fallback Engine**: Primary generation via Mistral AI (`mistral-large-latest`, `mistral-small-latest`), automated fallback to Google Gemini (`gemini-2.5-flash`), and deterministic `MockLLMProvider` for offline testing.

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone repository
cd ai-academic-engine

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your Primary Mistral API and Fallback Gemini API keys:
```env
# Primary LLM: Mistral AI
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL_NAME=mistral-large-latest

# Fallback LLM: Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```
If API keys are not provided or set to mock keys, the engine automatically uses the high-performance deterministic `MockLLMProvider`.

---

## 🧪 Running Tests

Execute the complete test suite:

```bash
python -m pytest -v
```

---

## 🌐 Running FastAPI Server

Start the API server using Uvicorn:

```bash
python -m uvicorn app.api.main:app --reload --port 8000
```

Access API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📡 REST API Endpoints

- `GET /api/v1/health`: System status healthcheck
- `GET /api/v1/board-profiles`: List versioned board profiles (CBSE, ICSE, UP Board)
- `POST /api/v1/exam/generate`: Generate complete exam paper with answer key
- `POST /api/v1/question/generate`: Generate individual question from blueprint
- `POST /api/v1/question/regenerate`: Targeted single-question regeneration
- `POST /api/v1/assignment/generate`: Generate practice assignment with rubric
- `POST /api/v1/curriculum/index`: Index raw curriculum text into RAG store

---

## 📜 License

Production Software - All rights reserved.
