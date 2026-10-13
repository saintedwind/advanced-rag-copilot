"""Data transfer objects and Pydantic validation models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    source_dir: Optional[str] = Field(default="data/sample_docs", description="Directory path containing documents to ingest")
    chunk_size: Optional[int] = Field(default=600, ge=100, le=4000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=1000)


class IngestionResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    collection_name: str
    elapsed_seconds: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, example="What are the key SLA tiers and guaranteed uptimes?")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of final context chunks after reranking")
    enable_reranking: bool = Field(default=True)
    use_agentic_flow: bool = Field(default=True, description="Whether to invoke LangGraph self-corrective workflow")


class CitationItem(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem] = []
    confidence: Optional[str] = "high"
    route_taken: Optional[str] = "retrieval_synthesized"
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_db_connected: bool
    reranker_ready: bool
