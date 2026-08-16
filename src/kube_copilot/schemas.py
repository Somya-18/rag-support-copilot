from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    title: str
    corpus_id: str
    chunks: int
    product_version: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    product_version: str = "latest"


class SourceResult(BaseModel):
    citation: int
    title: str
    heading: str
    excerpt: str
    score: float
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    hit_count: int
