# 🏗️ **ARCHITECTURE DOCUMENTATION**

## 📋 **System Overview**

DocuMind is a RAG (Retrieval-Augmented Generation) system that enables intelligent Q&A over uploaded documents. The system processes documents, extracts text, chunks them, generates embeddings, and provides contextual answers with citations.

## 🔄 **System Architecture Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   External      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   APIs          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Upload   │    │ Document        │    │ OpenAI API      │
│   Interface     │    │ Processing      │    │ (Embeddings)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Session       │    │ Chunking &      │    │ OpenAI API      │
│   Management    │    │ Embedding       │    │ (GPT-4o-mini)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │ Vector Search   │    │ Response        │
│   Viewer        │    │ & Retrieval     │    │ Generation      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 **Detailed Component Flow**

### **1. Document Ingestion Flow**
```
User Upload → File Validation → Text Extraction → Chunking → Embedding Generation → Storage
     │              │                │            │              │                    │
     ▼              ▼                ▼            ▼              ▼                    ▼
  Frontend    Type/Size Check   PDF/DOCX/TXT   Token-based   OpenAI API        In-memory +
  Interface                    Image OCR       Chunking      text-embedding-3   Pickle File
```

### **2. Q&A Flow**
```
User Question → Scope Selection → Query Embedding → Similarity Search → Context Assembly → LLM Generation → Response + Citations
      │              │                │                │                │                │                │
      ▼              ▼                ▼                ▼                ▼                ▼                ▼
   Frontend    All/Single Doc    OpenAI API      Cosine Similarity   Top-k Chunks    GPT-4o-mini     Formatted Answer
   Interface                    text-embedding-3   (numpy)           Assembly         Generation      with Sources
```

## 🔧 **Chunking Strategy**

### **Chunking Parameters**
- **Chunk Size**: 800 tokens
- **Overlap**: 150 tokens
- **Tokenizer**: `tiktoken` (GPT-4 compatible)
- **Strategy**: Sliding window with overlap

### **Why These Values?**
- **800 tokens**: Optimal balance between context preservation and embedding efficiency
- **150 tokens overlap**: Ensures no information loss at chunk boundaries
- **GPT-4 tokenizer**: Ensures compatibility with OpenAI models

### **Chunking Algorithm**
```python
def create_chunks(text, chunk_size=800, overlap=150):
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tiktoken.get_encoding("cl100k_base").decode(chunk_tokens)
        chunks.append({
            "content": chunk_text,
            "tokens": len(chunk_tokens),
            "start_token": i,
            "end_token": i + len(chunk_tokens)
        })
    
    return chunks
```

## 📋 **Metadata Schema**

### **Document Metadata**
```json
{
  "id": "uuid-string",
  "filename": "document.pdf",
  "file_type": ".pdf",
  "file_size": 1024000,
  "status": "indexed|processing|failed",
  "created_at": "2024-01-01T12:00:00Z",
  "session_id": "uuid-string",
  "chunk_count": 15,
  "total_tokens": 12000,
  "processing_time": 2.5,
  "error_message": null
}
```

### **Chunk Metadata**
```json
{
  "id": "uuid-string",
  "document_id": "uuid-string",
  "session_id": "uuid-string",
  "content": "chunk text content...",
  "chunk_index": 0,
  "page_number": 1,
  "token_count": 800,
  "start_token": 0,
  "end_token": 800,
  "embedding": [0.1, 0.2, ...],
  "created_at": "2024-01-01T12:00:00Z"
}
```

### **Session Metadata**
```json
{
  "id": "uuid-string",
  "created_at": "2024-01-01T12:00:00Z",
  "last_activity": "2024-01-01T12:30:00Z",
  "document_count": 5,
  "total_chunks": 75,
  "total_queries": 12
}
```

## 🗄️ **Data Storage Architecture**

### **In-Memory Storage (Development)**
```python
# Global data structures
documents = {}  # {session_id: {doc_id: doc_data}}
chunks = []     # Global chunks with session_id
sessions = {}   # {session_id: session_data}
```

### **Persistence Layer**
```python
# Pickle-based persistence
DATA_FILE = "data_simple.pkl"

def save_data():
    data = {
        "documents": documents,
        "chunks": chunks,
        "sessions": sessions
    }
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
            return data.get("documents", {}), data.get("chunks", []), data.get("sessions", {})
    return {}, [], {}
```

## 🔍 **Search & Retrieval Architecture**

### **Vector Search Implementation**
```python
def similarity_search(query_embedding, top_k=5):
    similarities = []
    for chunk in chunks:
        if chunk.get("session_id") == current_session_id:
            similarity = cosine_similarity(query_embedding, chunk["embedding"])
            similarities.append((similarity, chunk))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]
```

### **Cosine Similarity**
```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

## 🔐 **Session Management Architecture**

### **Session Isolation**
- Each user gets a unique session ID via cookies
- Documents and chunks are isolated per session
- Sessions expire after 24 hours of inactivity
- Automatic cleanup of expired sessions

### **Session Flow**
```
User Request → Cookie Check → Session Creation/Retrieval → Data Isolation → Response
      │              │                │                        │              │
      ▼              ▼                ▼                        ▼              ▼
   Frontend    Session ID      New/Existing Session    Filter by Session    User Data
   Request     Extraction      Creation/Retrieval      ID                   Only
```

## 🚀 **Performance Optimizations**

### **Caching Strategy**
- Embedding cache for repeated text
- Query result cache for similar questions
- LRU eviction policy
- Configurable cache sizes

### **Batch Processing**
- Background embedding generation
- Queue-based processing
- Timeout and retry mechanisms

### **Cost Optimization**
- Token usage tracking
- Efficient chunking to minimize API calls
- Cost estimation and alerts

## 🔧 **Error Handling Architecture**

### **Error Categories**
1. **File Processing Errors**: Invalid files, OCR failures
2. **API Errors**: Rate limits, network issues
3. **Embedding Errors**: Token limits, API failures
4. **Storage Errors**: Disk space, corruption

### **Error Recovery**
- Graceful degradation
- Retry mechanisms with exponential backoff
- User-friendly error messages
- Fallback strategies

## 📈 **Monitoring & Analytics**

### **Metrics Tracked**
- Document processing times
- Query response times
- API usage and costs
- Error rates and types
- User engagement metrics

### **Performance Monitoring**
- P95/P99 response times
- Throughput measurements
- Cache hit rates
- Memory usage

## 🔄 **Deployment Architecture**

### **Development Setup**
```
Frontend (Port 3001) ←→ Backend (Port 8001) ←→ OpenAI API
     │                       │                       │
     │                       │                       │
  Next.js                FastAPI                External
  Development            Development            API Calls
```

### **Production Considerations**
- Load balancing for multiple users
- Database persistence (PostgreSQL/MongoDB)
- Redis for caching
- Docker containerization
- Environment-based configuration

## 🛡️ **Security Architecture**

### **Input Validation**
- File type allowlist
- File size limits
- Content sanitization
- Session validation

### **API Security**
- Rate limiting
- Input sanitization
- Error message sanitization
- CORS configuration

## 📊 **Scalability Considerations**

### **Horizontal Scaling**
- Stateless backend design
- Shared storage layer
- Load balancer support
- Session management scaling

### **Vertical Scaling**
- Memory optimization
- CPU utilization
- I/O optimization
- Cache efficiency

This architecture provides a solid foundation for a production-ready RAG system with room for future enhancements and scaling.
