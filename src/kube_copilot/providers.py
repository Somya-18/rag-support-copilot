import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings


@dataclass
class ModelResult:
    data: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace = text.find("{")
    if brace != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, brace)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("Model did not return a valid JSON object")


class ModelProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def structured(self, instructions: str, payload: dict) -> ModelResult: ...


class OpenAIProvider(ModelProvider):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when PROVIDER_MODE=openai")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dimensions,
        )
        return [row.embedding for row in response.data]

    def structured(self, instructions: str, payload: dict) -> ModelResult:
        from openai import OpenAI
        response = self.client.chat.completions.create(
            model=self.settings.openai_generation_model,
            messages=[
                {"role": "system", "content": instructions + "\nReturn exactly one valid JSON object. Do not wrap it in markdown fences."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        usage = response.usage
        return ModelResult(
            data=_extract_json(response.choices[0].message.content or "{}"),
            model=response.model,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        )


_GEMINI_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class GeminiProvider(ModelProvider):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when PROVIDER_MODE=gemini")
        from google import genai
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    def structured(self, instructions: str, payload: dict) -> ModelResult:
        from google.genai import types as genai_types
        prompt = (
            instructions
            + "\nReturn exactly one valid JSON object. Do not wrap it in markdown fences.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_GEMINI_ANSWER_SCHEMA,
        )
        response = self._client.models.generate_content(
            model=self.settings.gemini_generation_model,
            contents=prompt,
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        return ModelResult(
            data=_extract_json(response.text),
            model=self.settings.gemini_generation_model,
            usage={
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            },
        )


def create_provider(settings: Settings | None = None) -> ModelProvider:
    cfg = settings or get_settings()
    if cfg.provider_mode == "gemini":
        return GeminiProvider(cfg)
    return OpenAIProvider(cfg)
