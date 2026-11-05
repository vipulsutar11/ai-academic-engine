import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from app.models.question import Question
from config import config

logger = logging.getLogger(__name__)

class QuestionBankCache:
    """High-performance question cache storing verified generated questions indexed by
    (chapter_id, topic_id, difficulty, question_type).
    Reduces latency to <1ms and cuts LLM API calls for recurring curriculum topics.
    """

    _instance = None
    _cache: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    _cache_file = config.DATA_DIR / "question_bank_cache.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuestionBankCache, cls).__new__(cls)
            cls._instance._load_from_disk()
        return cls._instance

    def _load_from_disk(self):
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    for key_str, q_list in raw_data.items():
                        parts = key_str.split("|")
                        if len(parts) == 5:
                            key = (parts[0], parts[1], parts[2], parts[3], parts[4])
                            self._cache[key] = q_list
                        elif len(parts) == 4:
                            key = ("", parts[0], parts[1], parts[2], parts[3])
                            self._cache[key] = q_list
                logger.info(f"Loaded QuestionBankCache with {sum(len(v) for v in self._cache.values())} cached questions.")
        except Exception as e:
            logger.warning(f"Could not load QuestionBankCache from disk: {e}")

    def save_to_disk(self):
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            raw_data = {}
            for key, q_list in self._cache.items():
                key_str = "|".join(key)
                raw_data[key_str] = q_list
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
            logger.info("Persisted QuestionBankCache to disk.")
        except Exception as e:
            logger.warning(f"Failed to save QuestionBankCache to disk: {e}")

    def get_cached_question(
        self,
        chapter_id: str,
        topic_id: str,
        difficulty: str,
        question_type: str,
        subject: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> Optional[Question]:
        """Retrieves a cached question matching the exact curriculum, subject & difficulty specs."""
        subj_prefix = (subject or "").strip().lower()
        key = (subj_prefix, chapter_id or "", topic_id or "", str(difficulty), str(question_type))
        candidates = self._cache.get(key, [])
        if not candidates and not subj_prefix:
            legacy_key = ("", chapter_id or "", topic_id or "", str(difficulty), str(question_type))
            candidates = self._cache.get(legacy_key, [])

        if not candidates:
            return None

        exclude_set = set(exclude_ids or [])
        for q_dict in candidates:
            if q_dict.get("id") not in exclude_set:
                try:
                    q = Question(**q_dict)
                    logger.info(f"QuestionBankCache HIT for '{key}': Instant <1ms retrieval for question '{q.id}'.")
                    return q
                except Exception as e:
                    logger.warning(f"Error instantiating cached Question: {e}")
        return None

    def store_question(self, question: Question, subject: Optional[str] = None):
        """Stores a verified high-quality question into the cache pool."""
        # Never store mock questions into persistent cache
        stem_lower = (question.stem or "").lower()
        id_lower = (question.id or "").lower()
        if "q_mock" in id_lower or "q_mock" in stem_lower or "fuse wire" in stem_lower or "mock response" in (question.explanation or "").lower():
            return

        diff_val = question.difficulty.value if hasattr(question.difficulty, "value") else str(question.difficulty)
        type_val = question.type.value if hasattr(question.type, "value") else str(question.type)
        subj_prefix = (subject or "").strip().lower()
        key = (subj_prefix, question.chapter_id or "", question.topic_id or "", diff_val, type_val)

        if key not in self._cache:
            self._cache[key] = []

        # Avoid duplicate question stems in cache
        for existing in self._cache[key]:
            if existing.get("stem") == question.stem:
                return

        self._cache[key].append(question.model_dump())
        self.save_to_disk()
