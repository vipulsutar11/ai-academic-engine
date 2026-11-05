import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).parent.resolve()

class Config:
    # App
    APP_NAME: str = "AI Academic Content & Examination Agent"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Primary LLM: Mistral AI (https://console.mistral.ai/)
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL_NAME: str = os.getenv("MISTRAL_MODEL_NAME", "open-mistral-7b")
    MISTRAL_API_BASE: str = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
    
    # Fallback LLM: Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    
    # LLM Settings
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # RAG Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "memory")  # 'faiss' or 'memory'
    VECTOR_STORE_DIR: Path = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "2"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Content & Quality Validation
    DUPLICATE_SIMILARITY_THRESHOLD: float = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.85"))
    QUALITY_PASS_THRESHOLD: float = float(os.getenv("QUALITY_PASS_THRESHOLD", "0.75"))
    
    # Data Paths
    DATA_DIR: Path = BASE_DIR / "data"
    CURRICULUM_DIR: Path = DATA_DIR / "curriculum"
    BLUEPRINTS_DIR: Path = DATA_DIR / "blueprints"

config = Config()
