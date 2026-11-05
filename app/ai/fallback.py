import logging
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from app.ai.provider import BaseLLMProvider
from app.ai.mistral import MistralProvider
from app.ai.gemini import GeminiProvider
from app.ai.mock_provider import MockLLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class TwoTierFallbackProvider(BaseLLMProvider):
    """Resilience wrapper with Mistral as Primary, Gemini as Fallback, and Mock Engine as emergency safety net:
    1. Primary Provider: Mistral AI Cloud API
    2. Fallback Tier 1: Google Gemini API
    3. Fallback Tier 2: Emergency Deterministic Rule-Based Engine (Mock Provider)
    """

    def __init__(
        self,
        primary_provider: Optional[BaseLLMProvider] = None,
        secondary_provider: Optional[BaseLLMProvider] = None,
        tertiary_provider: Optional[BaseLLMProvider] = None
    ):
        self.primary = primary_provider or MistralProvider()
        self.secondary = secondary_provider or GeminiProvider()
        self.tertiary = tertiary_provider or MockLLMProvider()

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> T:
        # Tier 1: Primary Mistral LLM
        try:
            return self.primary.generate_structured(prompt, response_schema, system_prompt, temperature)
        except Exception as primary_err:
            logger.warning(f"Primary Mistral LLM failed: {primary_err}. Initiating Fallback (Gemini API)...")

        # Tier 2: Fallback Gemini LLM
        try:
            return self.secondary.generate_structured(prompt, response_schema, system_prompt, temperature)
        except Exception as secondary_err:
            logger.warning(f"Fallback Gemini LLM failed: {secondary_err}. Initiating Emergency Fallback (Deterministic Rule-Based Engine)...")

        # Tier 3: Emergency Rule-Based Generator
        return self.tertiary.generate_structured(prompt, response_schema, system_prompt, temperature)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        # Tier 1: Primary Mistral LLM
        try:
            return self.primary.generate_text(prompt, system_prompt, temperature)
        except Exception as primary_err:
            logger.warning(f"Primary Mistral LLM text generation failed: {primary_err}. Initiating Fallback (Gemini API)...")

        # Tier 2: Fallback Gemini LLM
        try:
            return self.secondary.generate_text(prompt, system_prompt, temperature)
        except Exception as secondary_err:
            logger.warning(f"Fallback Gemini LLM text generation failed: {secondary_err}. Initiating Emergency Fallback (Deterministic Rule-Based Engine)...")

        # Tier 3: Emergency Rule-Based Generator
        return self.tertiary.generate_text(prompt, system_prompt, temperature)
