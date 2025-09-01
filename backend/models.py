# Data models for the RAG system

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Document(BaseModel):
    id: str
    filename: str
    file_size: int
    file_type: str
    upload_timestamp: str
    processing_status: str
    total_chunks: int
    session_id: str

class Chunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    tokens: int
    embedding: List[float]
    created_at: str

class ChatMessage(BaseModel):
    id: str
    session_id: str
    document_id: Optional[str]
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    timestamp: str
    processing_time: float

class Session(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    document_count: int

class Analytics(BaseModel):
    total_queries: int
    avg_query_time: float
    processing_times: List[float]
    P50: float
    P95: float
    P99: float
