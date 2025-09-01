# Architecture

## System Overview

The RAG Document Q&A system follows a microservices architecture with a FastAPI backend and Next.js frontend.

## Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │    │   Backend   │    │   OpenAI    │    │   Storage   │
│  (Next.js)  │    │  (FastAPI)  │    │     API     │    │  (Pickle)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       │ 1. Upload File    │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │ 2. Extract Text   │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │ 3. Chunk Text     │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │ 4. Generate       │                   │
       │                   │    Embeddings     │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │ 5. Store Chunks   │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │ 6. Ask Question   │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │ 7. Search         │                   │
       │                   │    Similar Chunks │                   │
       │                   │◀──────────────────│                   │
       │                   │                   │                   │
       │                   │ 8. Generate       │                   │
       │                   │    Answer         │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │ 9. Return Answer  │                   │                   │
       │◀──────────────────│                   │                   │
```

## Chunking Strategy

### Chunk Size: 800 tokens
- **Reasoning**: Optimal balance between context and granularity
- **Overlap**: 150-200 tokens between chunks
- **Benefits**: 
  - Maintains semantic coherence
  - Allows for precise retrieval
  - Reduces embedding costs

### Chunking Process:
1. **Text Extraction**: Extract raw text from documents
2. **Sentence Splitting**: Split into sentences using natural language boundaries
3. **Token Counting**: Count tokens using tiktoken
4. **Chunk Assembly**: Combine sentences to reach ~800 tokens
5. **Overlap Addition**: Add 150-200 token overlap between chunks

## Metadata Schema

### Document Metadata:
```json
{
  "id": "uuid-string",
  "filename": "original_filename.pdf",
  "file_size": 1024000,
  "file_type": "pdf",
  "upload_timestamp": "2025-01-09T12:00:00Z",
  "processing_status": "completed",
  "total_chunks": 25,
  "session_id": "session-uuid"
}
```

### Chunk Metadata:
```json
{
  "id": "chunk-uuid",
  "document_id": "document-uuid",
  "chunk_index": 0,
  "content": "chunk text content...",
  "tokens": 800,
  "page_number": 1,
  "start_position": 0,
  "end_position": 3200,
  "embedding_vector": [0.1, 0.2, ...],
  "created_at": "2025-01-09T12:00:00Z"
}
```

### Session Metadata:
```json
{
  "session_id": "session-uuid",
  "created_at": "2025-01-09T12:00:00Z",
  "last_activity": "2025-01-09T12:30:00Z",
  "document_count": 3,
  "total_queries": 15
}
```

### Chat History Metadata:
```json
{
  "id": "chat-uuid",
  "session_id": "session-uuid",
  "document_id": "document-uuid",
  "question": "What is the main topic?",
  "answer": "The main topic is...",
  "citations": [
    {
      "chunk_id": "chunk-uuid",
      "text": "cited text...",
      "page": 1,
      "source": "document.pdf",
      "similarity": 0.85
    }
  ],
  "timestamp": "2025-01-09T12:30:00Z",
  "processing_time": 2.5
}
```

## Technology Stack

### Backend:
- **Framework**: FastAPI (Python 3.13)
- **AI/ML**: OpenAI GPT-4o-mini, OpenAI Embeddings
- **Document Processing**: PyPDF2, python-docx, pytesseract, easyocr
- **Storage**: In-memory with pickle persistence
- **Deployment**: Render.com

### Frontend:
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React hooks
- **Real-time**: Server-Sent Events (SSE)
- **Deployment**: Render.com Static Site

### Key Design Decisions:

1. **Session-based Storage**: Each user session is isolated for security
2. **In-memory Storage**: Fast access for document chunks and embeddings
3. **Pickle Persistence**: Simple file-based persistence for session data
4. **Streaming Responses**: Real-time answer generation using SSE
5. **File Validation**: Security-first approach with content validation
6. **Progress Tracking**: Real-time processing status updates

## Performance Characteristics

- **Document Processing**: ~2-5 seconds per MB
- **Query Response**: <3 seconds for small corpora
- **Embedding Generation**: ~1 second per chunk
- **Memory Usage**: ~50MB per 1000 chunks
- **Concurrent Users**: 10-50 users per instance
