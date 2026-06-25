"""
Pydantic data models for CiteQuest-RAG.

These schemas are the type contract shared across all modules:
- ingestion produces Paper and Chunk objects
- retrieval consumes them and produces SearchResult objects
- API layer receives SearchRequest and returns SearchResponse
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models (ingestion output)
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    """A single academic paper with its metadata."""

    paper_id: str = Field(..., description="Stable unique ID (e.g. OpenAlex W-identifier)")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(default="", description="Reconstructed abstract text (may be empty)")
    year: int | None = Field(default=None, description="Publication year")
    venue: str | None = Field(default=None, description="Journal or conference name")
    authors: list[str] = Field(default_factory=list, description="Author display names in order")
    concepts: list[str] = Field(default_factory=list, description="Top-level concept labels")
    doi: str | None = Field(default=None, description="DOI identifier")
    url: str | None = Field(default=None, description="Landing page URL")
    citation_count: int = Field(default=0, ge=0, description="Number of citations")
    open_access: bool = Field(default=False, description="Whether the paper is open-access")


class Chunk(BaseModel):
    """A searchable text segment belonging to a paper."""

    chunk_id: str = Field(..., description="Unique chunk ID (e.g. '<paper_id>_default')")
    paper_id: str = Field(..., description="FK to papers.paper_id")
    chunk_text: str = Field(..., description="Full text of this chunk for indexing and display")
    chunk_type: Literal["metadata", "abstract", "full_text"] = Field(
        default="metadata",
        description="Source of this chunk text",
    )
    token_count: int = Field(default=0, ge=0, description="Approximate token count")


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Incoming search request (POST /search)."""

    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    mode: Literal["lexical", "vector", "hybrid"] = Field(
        default="lexical",
        description="Search mode — only 'lexical' is supported in v0.1",
    )
    year_from: int | None = Field(default=None, description="Optional: earliest publication year")
    year_to: int | None = Field(default=None, description="Optional: latest publication year")


class SearchResult(BaseModel):
    """A single search result."""

    paper_id: str
    chunk_id: str
    title: str
    year: int | None
    venue: str | None
    authors: list[str] = Field(default_factory=list)
    score: float = Field(..., description="Relevance score (higher is better)")
    snippet: str = Field(default="", description="Relevant text excerpt from the chunk")


class SearchResponse(BaseModel):
    """Complete search response (POST /search)."""

    query: str
    mode: str
    total_results: int
    results: list[SearchResult]
    latency_ms: float = Field(default=0.0, description="Server-side processing time in milliseconds")
