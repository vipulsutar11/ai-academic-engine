import pytest
from app.ai.mock_provider import MockLLMProvider
from app.models.question import Question
from app.models.exam import ExamPaper

from app.ai.fallback import TwoTierFallbackProvider
from app.ai.provider import BaseLLMProvider

class FailingProvider(BaseLLMProvider):
    def generate_structured(self, prompt, response_schema, system_prompt=None, temperature=0.2):
        raise RuntimeError("Cloud LLM network error / rate limit")

    def generate_text(self, prompt, system_prompt=None, temperature=0.2):
        raise RuntimeError("Cloud LLM network error")

def test_mock_provider_question_generation():
    provider = MockLLMProvider()
    q = provider.generate_structured(
        prompt="Generate a 3 mark short answer question on Ohm's Law",
        response_schema=Question
    )
    assert isinstance(q, Question)
    assert q.stem is not None
    assert q.marks > 0

def test_mock_provider_exam_generation():
    provider = MockLLMProvider()
    paper = provider.generate_structured(
        prompt="Generate full exam paper for Class 10 Science",
        response_schema=ExamPaper
    )
    assert isinstance(paper, ExamPaper)
    assert len(paper.sections) > 0

def test_two_tier_fallback_execution():
    failing_primary = FailingProvider()
    failing_secondary = FailingProvider()
    tertiary_mock = MockLLMProvider()

    fallback_provider = TwoTierFallbackProvider(
        primary_provider=failing_primary,
        secondary_provider=failing_secondary,
        tertiary_provider=tertiary_mock
    )

    q = fallback_provider.generate_structured(
        prompt="Generate a question on Newton's laws",
        response_schema=Question
    )
    assert isinstance(q, Question)
    assert q.stem is not None

def test_mistral_provider_initialization():
    from app.ai.mistral import MistralProvider
    provider = MistralProvider(api_key="mock-mistral-key")
    assert provider.api_key == "mock-mistral-key"
    assert provider.model_name in ["open-mistral-nemo", "open-mistral-7b", "mistral-large-latest", "mistral-small-latest"]
    assert "mistral.ai" in provider.api_base

def test_mistral_provider_custom_configuration():
    from app.ai.mistral import MistralProvider
    provider = MistralProvider(
        api_key="test-key",
        model_name="mistral-small-latest",
        api_base="https://custom.mistral.endpoint/v1"
    )
    assert provider.model_name == "mistral-small-latest"
    assert provider.api_base == "https://custom.mistral.endpoint/v1"
    assert provider.api_key == "test-key"

def test_pipeline_mistral_provider_default():
    from app.pipeline import AcademicContentPipeline
    from app.ai.mistral import MistralProvider
    # When initialized with MistralProvider directly
    mistral_inst = MistralProvider(api_key="test-key", model_name="mistral-large-latest")
    pipeline = AcademicContentPipeline(llm_provider=mistral_inst)
    assert pipeline.llm_provider is not None

def test_two_tier_fallback_defaults():
    from app.ai.fallback import TwoTierFallbackProvider
    from app.ai.mistral import MistralProvider
    from app.ai.gemini import GeminiProvider
    fallback_provider = TwoTierFallbackProvider()
    assert isinstance(fallback_provider.primary, MistralProvider)
    assert isinstance(fallback_provider.secondary, GeminiProvider)


