import json
import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
import openai
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

class MistralProvider(BaseLLMProvider):
    """Mistral AI LLM Provider implementation using Mistral's API endpoint.
    
    Compatible with Mistral models:
    - mistral-large-latest
    - mistral-small-latest
    - open-mistral-7b
    - codestral-latest
    - ministral-8b-latest
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        self.api_key = api_key or config.MISTRAL_API_KEY
        self.model_name = model_name or config.MISTRAL_MODEL_NAME
        self.api_base = api_base or config.MISTRAL_API_BASE

        self.client = openai.OpenAI(
            api_key=self.api_key or "mock-key",
            base_url=self.api_base
        )

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
        sys_msg += (
            f"\n\nCRITICAL OUTPUT REQUIREMENT:\n"
            f"You MUST return ONLY a valid, fully populated JSON instance matching this schema structure:\n"
            f"{json.dumps(schema_json, separators=(',', ':'))}\n\n"
            f"RULES:\n"
            f"1. DO NOT output the schema itself, do NOT output '$defs', and do NOT output 'properties'.\n"
            f"2. Fill in every required field with actual academic content (stem, options, correct_answer, marks, etc.).\n"
            f"3. Return ONLY the raw JSON object instance with no extra commentary or markdown."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=1200,
                timeout=config.LLM_TIMEOUT
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "response_format" in err_msg or "json_object" in err_msg or "unsupported" in err_msg:
                logger.info(f"Model '{self.model_name}' does not support response_format; retrying without it.")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    timeout=config.LLM_TIMEOUT
                )
            else:
                raise

        content = response.choices[0].message.content
        return parse_structured_json(content, response_schema)

    @with_retry(max_retries=config.MAX_RETRIES)
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        sys_msg = system_prompt or "You are an expert AI Academic Content Engine."

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            timeout=config.LLM_TIMEOUT
        )

        return response.choices[0].message.content.strip()
