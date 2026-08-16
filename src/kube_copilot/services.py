from functools import lru_cache

from .config import get_settings
from .db import SessionLocal
from .providers import ModelProvider, create_provider
from .retrieval import Retriever
from .workflow import RagPipeline


@lru_cache
def provider() -> ModelProvider:
    return create_provider(get_settings())


@lru_cache
def retriever() -> Retriever:
    return Retriever(SessionLocal, provider(), get_settings())


@lru_cache
def rag_pipeline() -> RagPipeline:
    return RagPipeline(provider(), retriever())
