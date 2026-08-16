import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .models import Chunk, Document
from .providers import ModelProvider


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    canonical_url: str
    product_version: str
    line_start: int
    line_end: int
    vector_score: float = 0.0
    lexical_score: float = 0.0
    fused_score: float = 0.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0 else 0.0


def _rrf(vector_hits: list[SearchHit], lexical_hits: list[SearchHit], k: int = 60) -> list[SearchHit]:
    merged: dict[str, SearchHit] = {}
    for source in (vector_hits, lexical_hits):
        for rank, hit in enumerate(source, start=1):
            key = str(hit.chunk_id)
            if key not in merged:
                merged[key] = hit
            merged[key].fused_score += 1 / (k + rank)
    return sorted(merged.values(), key=lambda h: h.fused_score, reverse=True)


class Retriever:
    def __init__(self, session_factory: Callable[[], Session], provider: ModelProvider, settings: Settings | None = None):
        self.session_factory = session_factory
        self.provider = provider
        self.settings = settings or get_settings()

    @property
    def _is_sqlite(self) -> bool:
        return self.settings.database_url.startswith("sqlite")

    def search(self, query: str, product_version: str = "latest") -> list[SearchHit]:
        vector = self.provider.embed([query])[0]
        vector_hits = self._vector_search(vector, product_version)
        lexical_hits = self._lexical_search(query, product_version)
        fused = _rrf(vector_hits, lexical_hits)[: self.settings.retrieval_fused_k]
        return fused[: self.settings.retrieval_final_k]

    def _base_filter(self, stmt, product_version: str):
        return stmt.where(
            Document.active.is_(True),
            or_(Document.product_version == product_version, Document.product_version == "latest"),
        )

    def _vector_search(self, vector: list[float], product_version: str) -> list[SearchHit]:
        if self._is_sqlite:
            return self._vector_search_sqlite(vector, product_version)
        distance = Chunk.embedding.op("<=>")(vector).label("distance")
        stmt = select(Chunk, Document, distance).join(Document).order_by(distance).limit(self.settings.retrieval_vector_k)
        stmt = self._base_filter(stmt, product_version)
        with self.session_factory() as db:
            rows = db.execute(stmt).all()
        return [self._hit(c, d, vector_score=max(0.0, 1 - float(dist))) for c, d, dist in rows]

    def _vector_search_sqlite(self, vector: list[float], product_version: str) -> list[SearchHit]:
        stmt = select(Chunk, Document).join(Document)
        stmt = self._base_filter(stmt, product_version)
        with self.session_factory() as db:
            rows = db.execute(stmt).all()
        scored = [(c, d, _cosine_similarity(vector, c.embedding)) for c, d in rows if c.embedding]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [self._hit(c, d, vector_score=max(0.0, s)) for c, d, s in scored[: self.settings.retrieval_vector_k]]

    def _lexical_search(self, query: str, product_version: str) -> list[SearchHit]:
        if self._is_sqlite:
            return self._lexical_search_sqlite(query, product_version)
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")
        stmt = (
            select(Chunk, Document, rank)
            .join(Document)
            .where(Chunk.search_vector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(self.settings.retrieval_lexical_k)
        )
        stmt = self._base_filter(stmt, product_version)
        with self.session_factory() as db:
            rows = db.execute(stmt).all()
        return [self._hit(c, d, lexical_score=float(s)) for c, d, s in rows]

    def _lexical_search_sqlite(self, query: str, product_version: str) -> list[SearchHit]:
        keywords = re.findall(r"\w+", query.lower())
        stmt = select(Chunk, Document).join(Document)
        stmt = self._base_filter(stmt, product_version)
        with self.session_factory() as db:
            rows = db.execute(stmt).all()
        scored = []
        for chunk, doc in rows:
            text_lower = (chunk.content or "").lower()
            score = sum(1.0 / (i + 1) for i, kw in enumerate(keywords) if kw in text_lower)
            if score > 0:
                scored.append((chunk, doc, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [self._hit(c, d, lexical_score=s) for c, d, s in scored[: self.settings.retrieval_lexical_k]]

    @staticmethod
    def _hit(chunk: Chunk, doc: Document, **scores) -> SearchHit:
        return SearchHit(
            chunk_id=str(chunk.id),
            document_id=str(doc.id),
            title=doc.title,
            heading_path=chunk.heading_path or [],
            content=chunk.content,
            canonical_url=doc.canonical_url,
            product_version=doc.product_version,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            **scores,
        )
