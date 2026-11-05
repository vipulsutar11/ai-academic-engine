import time
import json
import logging
import functools
from typing import Callable, Any, Type, TypeVar
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

def clean_json_text(text: str) -> str:
    """Fixes common markdown wrappers or formatting artifacts in raw LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def parse_structured_json(text: str, schema_class: Type[T]) -> T:
    """Parses text into Pydantic model with fallback repair and nested unwrapping."""
    cleaned = clean_json_text(text)
    data = None
    try:
        data = json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Initial JSON parse failed: {e}. Attempting extraction...")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            snippet = cleaned[start:end+1]
            try:
                data = json.loads(snippet)
            except Exception:
                raise e
        else:
            raise e

    # Reject accidental JSON schema echoes (contains '$defs' but missing populated fields)
    if isinstance(data, dict) and "$defs" in data and not any(k in data for k in ["stem", "sections", "questions"]):
        raise ValueError("Model returned JSON Schema definition instead of populated object instance.")

    # Auto-unwrap if model placed the result under an outer wrapper key
    if isinstance(data, dict):
        for wrapper_key in ["question", "data", "result", "exam_paper", "response", schema_class.__name__.lower()]:
            if wrapper_key in data and isinstance(data[wrapper_key], dict):
                try:
                    return schema_class.model_validate(data[wrapper_key])
                except Exception:
                    pass

    return schema_class.model_validate(data)

def with_retry(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator providing exponential backoff retries for LLM calls."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    err_str = str(e)
                    if "Quota exceeded" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        logger.warning(f"Quota limit reached. Aborting retries immediately: {e}")
                        raise e
                    logger.warning(f"LLM call attempt {attempt}/{max_retries} failed with error: {e}. Retrying in {delay:.1f}s...")
                    if attempt == max_retries:
                        break
                    time.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_exception}") from last_exception
        return wrapper
    return decorator
