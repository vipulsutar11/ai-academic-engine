import os
import json
import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
import google.generativeai as genai
from app.ai.provider import BaseLLMProvider
from app.ai.retry import with_retry, parse_structured_json
from config import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

def _clean_schema_dict(d):
    if isinstance(d, dict):
        return {k: _clean_schema_dict(v) for k, v in d.items() if k not in ("title", "description", "default")}
    elif isinstance(d, list):
        return [_clean_schema_dict(x) for x in d]
    return d

class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider implementation supporting structured output generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        raw_model = model_name or config.GEMINI_MODEL_NAME or "gemini-3.8-flash"
        if "2.5-flash" in raw_model or "1.5-flash" in raw_model:
            raw_model = "gemini-3.8-flash"
        self.model_name = raw_model
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.warning("GEMINI_API_KEY not set. Gemini API calls will require GEMINI_API_KEY in .env.")

    @with_retry(max_retries=config.MAX_RETRIES)
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> T:
        schema_json = _clean_schema_dict(response_schema.model_json_schema())
        
        sys_msg = system_prompt or "You are an expert AI Academic Content Engine."
        full_prompt = (
            f"{sys_msg}\n\n"
            f"User Request:\n{prompt}\n\n"
            f"CRITICAL REQUIREMENT: You MUST respond ONLY with a valid JSON object matching this schema:\n"
            f"{json.dumps(schema_json, separators=(',', ':'))}\n\n"
            f"Do not include any explanation or markdown formatting."
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json"
                )
            )
            content = response.text.strip()
            return parse_structured_json(content, response_schema)
        except Exception as e:
            logger.warning(f"Gemini API generation failed: {e}")
            raise RuntimeError(f"Gemini API error: {e}") from e

    @with_retry(max_retries=config.MAX_RETRIES)
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        sys_msg = system_prompt or "You are an expert AI Academic Content Engine."
        full_prompt = f"{sys_msg}\n\nUser Request:\n{prompt}"

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API text generation failed: {e}")
            raise RuntimeError(f"Gemini API error: {e}") from e
