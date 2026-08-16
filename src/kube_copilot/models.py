import json
import uuid as _uuid_module
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


# ── cross-dialect TypeDecorators ─────────────────────────────────────────────


class UuidType(TypeDecorator):
    impl = String
    cache_ok = True

    def __init__(self):
        super().__init__(36)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, _uuid_module.UUID):
            return value
        return _uuid_module.UUID(str(value))


class JsonType(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class ArrayStringType(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(String))
        return dialect.type_descriptor(JSON())


class VectorType(TypeDecorator):
    impl = Vector
    cache_ok = True

    def __init__(self, dimensions: int = 3072):
        self.dimensions = dimensions
        super().__init__(dimensions)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dimensions))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name != "postgresql" and isinstance(value, str):
            return json.loads(value)
        return value


class TsvectorType(TypeDecorator):
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(TSVECTOR())
        return dialect.type_descriptor(Text())


# ── helpers ───────────────────────────────────────────────────────────────────


def utcnow() -> datetime:
    return datetime.now(UTC)


# ── models ────────────────────────────────────────────────────────────────────


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_corpus_version", "corpus_id", "product_version", "document_type"),
    )
    id: Mapped[_uuid_module.UUID] = mapped_column(UuidType(), primary_key=True, default=_uuid_module.uuid4)
    corpus_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="public")
    access_tags: Mapped[list[str]] = mapped_column(ArrayStringType(), default=lambda: ["public"])
    product_version: Mapped[str] = mapped_column(String(32), index=True)
    source_path: Mapped[str] = mapped_column(String(1000), unique=True)
    canonical_url: Mapped[str] = mapped_column(String(1000))
    title: Mapped[str] = mapped_column(String(500))
    document_type: Mapped[str] = mapped_column(String(64), index=True)
    checksum: Mapped[str] = mapped_column(String(64))
    source_commit: Mapped[str] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    metadata_json: Mapped[dict] = mapped_column(JsonType(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_ordinal", "document_id", "ordinal", unique=True),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )
    id: Mapped[_uuid_module.UUID] = mapped_column(UuidType(), primary_key=True, default=_uuid_module.uuid4)
    document_id: Mapped[_uuid_module.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(ArrayStringType(), default=list)
    content: Mapped[str] = mapped_column(Text)
    embedded_text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(VectorType(3072))
    search_vector: Mapped[str | None] = mapped_column(TsvectorType())
    document: Mapped[Document] = relationship(back_populates="chunks")
