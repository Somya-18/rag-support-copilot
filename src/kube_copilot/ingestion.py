import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import yaml
from pypdf import PdfReader
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .models import Chunk, Document
from .providers import ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class ParsedChunk:
    ordinal: int
    heading_path: list[str]
    content: str
    embedded_text: str
    token_count: int
    line_start: int
    line_end: int


def _tokens(text_value: str) -> list[int]:
    return tiktoken.get_encoding("cl100k_base").encode(text_value)


def parse_front_matter(raw: str) -> tuple[dict, list[str], int]:
    lines = raw.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return yaml.safe_load("\n".join(lines[1:idx])) or {}, lines[idx + 1:], idx + 2
    return {}, lines, 1


def chunk_markdown(raw: str, source_path: str, target: int = 450, maximum: int = 700, overlap: int = 60) -> tuple[dict, list[ParsedChunk]]:
    meta, lines, line_offset = parse_front_matter(raw)
    title = str(meta.get("title") or Path(source_path).stem.replace("-", " ").title())
    headings: list[str] = []
    sections: list[tuple[list[str], list[tuple[int, str]]]] = []
    current: list[tuple[int, str]] = []
    in_fence = False
    for index, line in enumerate(lines, start=line_offset):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line) if not in_fence else None
        if match:
            if current:
                sections.append((headings.copy(), current))
                current = []
            level = len(match.group(1))
            headings = headings[: level - 1] + [re.sub(r"\s+\{#.*\}$", "", match.group(2))]
        else:
            current.append((index, line))
    if current:
        sections.append((headings.copy(), current))

    chunks: list[ParsedChunk] = []
    for heading_path, numbered in sections:
        paragraphs: list[list[tuple[int, str]]] = []
        paragraph: list[tuple[int, str]] = []
        in_fence = False
        for item in numbered:
            if item[1].lstrip().startswith("```"):
                in_fence = not in_fence
            if not item[1].strip() and paragraph and not in_fence:
                paragraphs.append(paragraph)
                paragraph = []
            else:
                paragraph.append(item)
        if paragraph:
            paragraphs.append(paragraph)
        window: list[tuple[int, str]] = []
        for para in paragraphs:
            proposed = window + ([(-1, "")] if window else []) + para
            proposed_text = "\n".join(value for _, value in proposed).strip()
            if window and len(_tokens(proposed_text)) > target:
                chunks.extend(_emit_window(title, heading_path, window, len(chunks), maximum))
                carry_text = "\n".join(value for _, value in window)
                carry_ids = _tokens(carry_text)[-overlap:]
                carry = tiktoken.get_encoding("cl100k_base").decode(carry_ids)
                window = [(window[-1][0], carry)] if carry.strip() else []
            window += para
        if window:
            chunks.extend(_emit_window(title, heading_path, window, len(chunks), maximum))
    for ordinal, chunk in enumerate(chunks):
        chunk.ordinal = ordinal
    return meta | {"title": title}, chunks


def _emit_window(title: str, headings: list[str], lines: list[tuple[int, str]], ordinal: int, maximum: int) -> list[ParsedChunk]:
    content = "\n".join(value for _, value in lines).strip()
    if not content:
        return []
    encoding = tiktoken.get_encoding("cl100k_base")
    token_ids = encoding.encode(content)
    results = []
    for start in range(0, len(token_ids), maximum):
        part = encoding.decode(token_ids[start: start + maximum]).strip()
        if not part:
            continue
        prefix = " > ".join([title, *headings])
        results.append(ParsedChunk(ordinal, headings, part, f"{prefix}\n{part}", len(_tokens(part)), lines[0][0], lines[-1][0]))
    return results


def _pdf_to_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


class IngestionService:
    def __init__(self, provider: ModelProvider, settings: Settings | None = None):
        self.provider = provider
        self.settings = settings or get_settings()

    def ingest_pdf(self, db: Session, file_bytes: bytes, filename: str, product_version: str = "latest") -> dict:
        text = _pdf_to_text(file_bytes)
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF — it may be scanned or image-only.")

        title = Path(filename).stem.replace("-", " ").replace("_", " ").title()
        front = yaml.dump({"title": title}, allow_unicode=True, default_flow_style=False)
        meta, parsed = chunk_markdown(f"---\n{front}---\n{text}", filename)

        logger.info("pdf %s → %d chunks, embedding...", filename, len(parsed))
        embeddings: list[list[float]] = []
        for offset in range(0, len(parsed), 64):
            embeddings.extend(self.provider.embed([c.embedded_text for c in parsed[offset: offset + 64]]))

        corpus_id = hashlib.sha256(file_bytes).hexdigest()
        is_sqlite = self.settings.database_url.startswith("sqlite")

        existing = db.execute(select(Document).where(Document.source_path == filename)).scalar_one_or_none()
        if existing:
            existing.chunks.clear()
            db.flush()

        doc = existing or Document(source_path=filename)
        doc.corpus_id = corpus_id
        doc.product_version = product_version
        doc.canonical_url = f"file://{filename}"
        doc.title = title
        doc.document_type = "documentation"
        doc.checksum = corpus_id
        doc.source_commit = corpus_id[:12]
        doc.metadata_json = meta
        doc.active = True

        if not existing:
            db.add(doc)
        db.flush()

        for item, embedding in zip(parsed, embeddings, strict=True):
            search_vec = item.embedded_text if is_sqlite else None
            db.add(Chunk(document_id=doc.id, embedding=embedding, search_vector=search_vec, **item.__dict__))

        if not is_sqlite:
            db.execute(text("UPDATE chunks SET search_vector = to_tsvector('english', embedded_text) WHERE search_vector IS NULL"))

        db.commit()
        logger.info("pdf %s ingested: %d chunks, corpus_id=%s", filename, len(parsed), corpus_id[:12])
        return {"filename": filename, "title": title, "corpus_id": corpus_id, "chunks": len(parsed), "product_version": product_version}
