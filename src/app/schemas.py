from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestionRequest(StrictModel):
    source_dir: str | None = None
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)

    @model_validator(mode="after")
    def overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        return self


class QueryRequest(StrictModel):
    query: str = Field(min_length=2, max_length=400)
    top_k: int = Field(default=4, ge=1, le=10)
    enable_reranking: bool = True
    use_agentic_flow: bool = True

    @field_validator("query")
    @classmethod
    def clean(cls, value):
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Question is too short")
        return value


class CitationItem(BaseModel):
    index: int
    source: str
    chunk_id: str
    snippet: str
    score: float | None = None
    source_sha256: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    start_line: int | None = None
    end_line: int | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    confidence: str
    status: str
    route_taken: str
    generation_mode: str
    elapsed_seconds: float
    cache_hit: bool = False
    usage: dict = Field(default_factory=dict)
    evidence_screen: dict = Field(default_factory=dict)
    corpus_sha256: str | None = None
