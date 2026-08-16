import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .ingestion import IngestionService
from .schemas import QueryRequest, QueryResponse, UploadResponse
from .services import provider, rag_pipeline

logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    from . import models as _models  # noqa: F401 — registers tables to Base.metadata
    from .db import ensure_schema
    ensure_schema()
    yield


app = FastAPI(
    title="Document Ingestion and RAG Service",
    version="0.2.0",
    description="Upload PDFs, embed them, and query across all ingested documents.",
    lifespan=lifespan,
)


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(503, "database unavailable") from exc


@app.post("/v1/ingestion/upload", response_model=UploadResponse, status_code=201)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document to ingest"),
    product_version: str = Form(default="latest"),
    db: Session = Depends(get_db),
):
    """Upload a PDF and immediately chunk, embed, and index its content."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are supported")
    content = await file.read()
    svc = IngestionService(provider(), get_settings())
    try:
        result = svc.ingest_pdf(db, content, file.filename, product_version)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result


@app.post("/v1/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    """Ask a question — retrieves relevant chunks and generates a cited answer."""
    result = rag_pipeline().query(payload.question, product_version=payload.product_version)
    return result
