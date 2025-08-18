from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import os
import uuid
from openai import OpenAI
import PyPDF2
import re
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from docx import Document as DocxDocument
# import tiktoken  # Commented out due to Rust compilation issues
import numpy as np
import time
from PIL import Image
import pytesseract
import easyocr
import io
import logging
from collections import defaultdict, Counter
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
if api_key == "your_openai_api_key_here":
    print("Please set your OpenAI API key!")
    print("Option 1: Set environment variable: export OPENAI_API_KEY='your_key_here'")
    print("Option 2: Edit this file and replace 'your_openai_api_key_here' with your actual key")
client = OpenAI(api_key=api_key)

app = FastAPI(title="Advanced RAG Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:3001",
        "https://rag-document-qa-iota.vercel.app",
        "https://rag-document-qa.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG Document Q&A Backend is running", "status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/debug")
async def debug():
    return {"message": "Debug endpoint working", "timestamp": datetime.now().isoformat()}

documents = {}
chunks = []
sessions = {}
DATA_FILE = "data_simple.pkl"

analytics = {
    "total_queries": 0,
    "total_documents": 0,
    "total_chunks": 0,
    "query_times": [],
    "popular_queries": Counter(),
    "document_types": Counter(),
    "error_counts": Counter(),
    "session_activity": defaultdict(int),
    "api_usage": {
        "embeddings_generated": 0,
        "tokens_processed": 0,
        "files_processed": 0
    }
}

shared_documents = {}
document_comments = {}

embedding_cache = {}
query_cache = {}

def get_session_id(request: Request) -> str:
    """Get or create session ID from cookies"""
    session_id = request.cookies.get("session_id")
    
    if not session_id or session_id not in sessions:
        # Create new session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    
    # Update last activity
    sessions[session_id]["last_activity"] = datetime.now().isoformat()
    analytics["session_activity"][session_id] += 1
    return session_id

def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    cutoff = datetime.now() - timedelta(hours=24)
    expired_sessions = []
    
    for session_id, session_data in sessions.items():
        last_activity = datetime.fromisoformat(session_data["last_activity"])
        if last_activity < cutoff:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del sessions[session_id]
        if session_id in documents:
            del documents[session_id]
    
    # Remove chunks for expired sessions
    global chunks
    chunks = [c for c in chunks if c.get("session_id") not in expired_sessions]

def save_data():
    """Save data to file with analytics"""
    data = {
        "documents": documents,
        "chunks": chunks,
        "sessions": sessions,
        "analytics": analytics,
        "shared_documents": shared_documents,
        "document_comments": document_comments
    }
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def load_data():
    """Load data from file"""
    global documents, chunks, sessions, analytics, shared_documents, document_comments
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
            documents = data.get("documents", {})
            chunks = data.get("chunks", [])
            sessions = data.get("sessions", {})
            analytics.update(data.get("analytics", {}))
            shared_documents = data.get("shared_documents", {})
            document_comments = data.get("document_comments", {})

# Load existing data
load_data()

class RAGProcessor:
    def __init__(self):
        # Simple token counting using word-based approach
        self.chunk_size = 800
        self.chunk_overlap = 150
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting using word-based approach"""
        # Rough approximation: 1 token ≈ 4 characters or 1 word
        return len(text.split())
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap using word-based approach"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if text_hash in embedding_cache:
            return embedding_cache[text_hash]
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding
            embedding_cache[text_hash] = embedding
            analytics["api_usage"]["embeddings_generated"] += 1
            analytics["api_usage"]["tokens_processed"] += len(text.split())
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

rag_processor = RAGProcessor()

def clean_markdown(text: str) -> str:
    """Remove markdown formatting"""
    # Remove asterisks, bold, italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove other markdown
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text.strip()

def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text from various file types"""
    try:
        if file_type.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        elif file_type.lower() == '.docx':
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        elif file_type.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_type.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return extract_text_from_image(file_path)
        else:
            raise Exception(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise Exception(f"Error extracting text from {file_type} file: {str(e)}")

def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_path)
        try:
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image_path)
            text = ""
            for (bbox, text_detected, confidence) in results:
                if confidence > 0.5:
                    text += text_detected + "\n"
            if text.strip():
                return text
        except Exception as e:
            print(f"EasyOCR failed, trying Tesseract: {e}")
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        except Exception as e:
            print(f"Tesseract failed: {e}")
            return "OCR failed to extract text from this image."
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def process_document_sync(file_id: str, file_path: str, filename: str, file_ext: str, session_id: str):
    """Process document synchronously with analytics"""
    try:
        # Update status
        documents[session_id][file_id]["status"] = "extracting"
        save_data()
        
        # Extract text
        text_content = extract_text_from_file(file_path, file_ext)
        
        # Update status
        documents[session_id][file_id]["status"] = "chunking"
        save_data()
        
        # Chunk text
        text_chunks = rag_processor.chunk_text(text_content)
        
        # Create embeddings and store chunks
        for i, chunk in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            
            # Get embedding
            embedding = rag_processor.get_embedding(chunk)
            
            # Create chunk record with metadata
            chunk_record = {
                "id": chunk_id,
                "doc_id": file_id,
                "doc_name": filename,
                "session_id": session_id,  # Add session isolation
                "content": chunk,
                "embedding": embedding,
                "chunk_index": i,
                "page": 1,
                "created_at": datetime.now().isoformat()
            }
            
            chunks.append(chunk_record)
        
        # Update document status
        documents[session_id][file_id]["status"] = "indexed"
        documents[session_id][file_id]["chunk_count"] = len(text_chunks)
        documents[session_id][file_id]["content"] = text_content
        documents[session_id][file_id]["file_type"] = file_ext
        
        # Update analytics
        analytics["total_documents"] += 1
        analytics["total_chunks"] += len(text_chunks)
        analytics["document_types"][file_ext] += 1
        analytics["api_usage"]["files_processed"] += 1
        
        save_data()
        
        return True
            
    except Exception as e:
        documents[session_id][file_id]["status"] = "failed"
        documents[session_id][file_id]["error"] = str(e)
        analytics["error_counts"]["document_processing"] += 1
        save_data()
        return False

@app.get("/api/health")
async def health_check():
    """Enhanced health check with analytics and API key validation"""
    cleanup_old_sessions()
    
    # Check API key status
    api_key_status = "valid"
    api_key_message = "API key is configured"
    
    if api_key == "your_openai_api_key_here":
        api_key_status = "error"
        api_key_message = "OpenAI API key not configured - please set your API key"
    else:
        try:
            # Test API key with a simple request
            client.models.list()
        except Exception as e:
            api_key_status = "error"
            api_key_message = f"API key validation failed: {str(e)}"
    
    # Calculate performance metrics
    avg_query_time = np.mean(analytics["query_times"]) if analytics["query_times"] else 0
    total_sessions = len(sessions)
    active_sessions = len([s for s in sessions.values() 
                          if datetime.fromisoformat(s["last_activity"]) > datetime.now() - timedelta(hours=1)])
    
    return {
        "status": "healthy" if api_key_status == "valid" else "error",
        "message": "Advanced RAG API is running",
        "api_key": {
            "status": api_key_status,
            "message": api_key_message
        },
        "documents_count": analytics["total_documents"],
        "chunks_count": analytics["total_chunks"],
        "sessions_count": total_sessions,
        "active_sessions": active_sessions,
        "analytics": {
            "total_queries": analytics["total_queries"],
            "avg_query_time": round(avg_query_time, 2),
            "popular_file_types": dict(analytics["document_types"].most_common(3)),
            "api_usage": analytics["api_usage"],
            "error_counts": dict(analytics["error_counts"])
        },
        "performance": {
            "embedding_cache_size": len(embedding_cache),
            "query_cache_size": len(query_cache),
            "memory_usage": "optimized"
        }
    }

@app.get("/api/analytics")
async def get_analytics(request: Request):
    """Get detailed analytics"""
    session_id = get_session_id(request)
    
    # User-specific analytics
    user_docs = len(documents.get(session_id, {}))
    user_queries = analytics["session_activity"].get(session_id, 0)
    
    return {
        "user_stats": {
            "documents_uploaded": user_docs,
            "queries_made": user_queries,
            "session_created": sessions[session_id]["created_at"]
        },
        "system_stats": {
            "total_documents": analytics["total_documents"],
            "total_queries": analytics["total_queries"],
            "total_chunks": analytics["total_chunks"],
            "avg_query_time": np.mean(analytics["query_times"]) if analytics["query_times"] else 0
        },
        "popular_queries": dict(analytics["popular_queries"].most_common(5)),
        "file_type_distribution": dict(analytics["document_types"]),
        "performance_metrics": {
            "cache_hit_rate": len(embedding_cache) / max(analytics["api_usage"]["embeddings_generated"], 1),
            "api_efficiency": analytics["api_usage"]["tokens_processed"] / max(analytics["api_usage"]["embeddings_generated"], 1)
        }
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):
    """Upload document with enhanced features"""
    session_id = get_session_id(request)
    
    # Initialize session documents if not exists
    if session_id not in documents:
        documents[session_id] = {}
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported")
    
    # Validate file size (30MB limit)
    max_size = 30 * 1024 * 1024  # 30MB
    if file.size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 30MB")
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Save file temporarily
    file_path = f"temp_{file_id}{file_ext}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize document record
        documents[session_id][file_id] = {
            "id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "created_at": datetime.now().isoformat(),
            "session_id": session_id,
            "file_size": len(content),
            "file_type": file_ext
        }
        save_data()
        
        # Process document
        success = process_document_sync(file_id, file_path, file.filename, file_ext, session_id)
        
        if success:
            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "indexed",
                "message": "File uploaded and processed successfully."
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process document")
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents(request: Request):
    """Get documents with collaboration features"""
    session_id = get_session_id(request)
    
    # Get user's own documents
    user_docs = documents.get(session_id, {})
    
    # Get shared documents
    shared_docs = []
    for doc_id, share_info in shared_documents.items():
        if session_id in share_info.get("shared_with", []):
            # Find the original document
            for owner_session, docs in documents.items():
                if doc_id in docs:
                    doc = docs[doc_id].copy()
                    doc["shared_by"] = share_info["shared_by"]
                    doc["permissions"] = share_info["permissions"]
                    shared_docs.append(doc)
                    break
    
    # Combine and format documents
    all_docs = []
    for doc in user_docs.values():
        doc_copy = doc.copy()
        doc_copy["ownership"] = "own"
        all_docs.append(doc_copy)
    
    for doc in shared_docs:
        doc_copy = doc.copy()
        doc_copy["ownership"] = "shared"
        all_docs.append(doc_copy)
    
    return all_docs

@app.post("/api/documents/{doc_id}/share")
async def share_document(doc_id: str, request: Request):
    """Share document with other sessions"""
    session_id = get_session_id(request)
    
    # Check if user owns the document
    if session_id not in documents or doc_id not in documents[session_id]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get share request data
    data = await request.json()
    target_session = data.get("session_id")
    permissions = data.get("permissions", "read")
    
    if not target_session:
        raise HTTPException(status_code=400, detail="Target session ID required")
    
    # Initialize sharing
    if doc_id not in shared_documents:
        shared_documents[doc_id] = {
            "shared_by": session_id,
            "shared_with": [],
            "permissions": "read"
        }
    
    # Add to shared list
    if target_session not in shared_documents[doc_id]["shared_with"]:
        shared_documents[doc_id]["shared_with"].append(target_session)
    
    save_data()
    
    return {"message": f"Document shared with session {target_session}"}

@app.post("/api/documents/{doc_id}/comment")
async def add_comment(doc_id: str, request: Request):
    """Add comment to document"""
    session_id = get_session_id(request)
    
    # Get comment data
    data = await request.json()
    comment_text = data.get("comment")
    
    if not comment_text:
        raise HTTPException(status_code=400, detail="Comment text required")
    
    # Initialize comments if not exists
    if doc_id not in document_comments:
        document_comments[doc_id] = []
    
    # Add comment
    comment = {
        "session_id": session_id,
        "comment": comment_text,
        "timestamp": datetime.now().isoformat()
    }
    
    document_comments[doc_id].append(comment)
    save_data()
    
    return {"message": "Comment added successfully"}

@app.get("/api/documents/{doc_id}/comments")
async def get_comments(doc_id: str, request: Request):
    """Get comments for document"""
    session_id = get_session_id(request)
    
    # Check access permissions
    has_access = False
    if session_id in documents and doc_id in documents[session_id]:
        has_access = True
    elif doc_id in shared_documents and session_id in shared_documents[doc_id]["shared_with"]:
        has_access = True
    
    if not has_access:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return document_comments.get(doc_id, [])

@app.get("/api/documents/{doc_id}/content")
async def get_document_content(doc_id: str, request: Request):
    """Get document content with access control"""
    session_id = get_session_id(request)
    
    # Check access permissions
    doc = None
    if session_id in documents and doc_id in documents[session_id]:
        doc = documents[session_id][doc_id]
    elif doc_id in shared_documents and session_id in shared_documents[doc_id]["shared_with"]:
        # Find shared document
        for owner_session, docs in documents.items():
            if doc_id in docs:
                doc = docs[doc_id]
                break
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "content": doc.get("content", ""),
        "chunk_count": doc.get("chunk_count", 0),
        "status": doc["status"],
        "created_at": doc["created_at"]
    }

@app.get("/api/documents/{doc_id}/status")
async def get_document_status(doc_id: str, request: Request):
    """Get document status"""
    session_id = get_session_id(request)
    
    if session_id not in documents or doc_id not in documents[session_id]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents[session_id][doc_id]
    return doc

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    """Delete document"""
    session_id = get_session_id(request)
    
    if session_id not in documents or doc_id not in documents[session_id]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove document
    del documents[session_id][doc_id]
    
    # Remove chunks
    global chunks
    chunks = [c for c in chunks if c["doc_id"] != doc_id]
    
    # Remove from shared documents
    if doc_id in shared_documents:
        del shared_documents[doc_id]
    
    # Remove comments
    if doc_id in document_comments:
        del document_comments[doc_id]
    
    save_data()
    
    return {"message": "Document deleted successfully"}

@app.post("/api/qa")
async def ask_question(request: Request):
    """Ask a question with enhanced analytics and caching"""
    session_id = get_session_id(request)
    
    # Get request data
    data = await request.json()
    query = data.get("question", "")
    doc_id = data.get("document_id")
    scope = data.get("scope", "all")
    
    if not query:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Update analytics
    analytics["total_queries"] += 1
    analytics["popular_queries"][query.lower()[:50]] += 1
    
    # Check query cache
    query_hash = hashlib.md5(f"{query}_{doc_id}_{scope}".encode()).hexdigest()
    if query_hash in query_cache:
        cached_result = query_cache[query_hash]
        cached_result["cached"] = True
        return cached_result
    
    # Filter documents by session
    session_docs = [
        doc for doc in documents.get(session_id, {}).values() 
        if doc.get("status") == "indexed"
    ]
    
    # Add shared documents
    for share_doc_id, share_info in shared_documents.items():
        if session_id in share_info.get("shared_with", []):
            for owner_session, docs in documents.items():
                if share_doc_id in docs and docs[share_doc_id]["status"] == "indexed":
                    session_docs.append(docs[share_doc_id])
                    break
    
    if not session_docs:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "citations": [],
            "sources": []
        }
    
    try:
        start_time = time.time()
        
        # Get query embedding
        query_embedding = rag_processor.get_embedding(query)
        if not query_embedding:
            return {
                "answer": "Error: Could not process query embedding.",
                "citations": [],
                "sources": []
            }
        
        # Filter chunks by document if specified
        candidate_chunks = [c for c in chunks if c.get("session_id") == session_id]
        
        # Add shared document chunks
        for share_doc_id, share_info in shared_documents.items():
            if session_id in share_info.get("shared_with", []):
                shared_chunks = [c for c in chunks if c["doc_id"] == share_doc_id]
                candidate_chunks.extend(shared_chunks)
        
        if scope == "document" and doc_id:
            candidate_chunks = [chunk for chunk in candidate_chunks if chunk["doc_id"] == doc_id]
        
        # Calculate similarities
        similarities = []
        for chunk in candidate_chunks:
            if chunk.get("embedding"):
                similarity = rag_processor.cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:8]  # Top 8 results
        
        if not top_results:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "citations": [],
                "sources": []
            }
        
        # Prepare context for LLM
        context_chunks = []
        relevant_chunks = []
        
        for similarity, chunk in top_results:
            context_chunks.append(f"Document: {chunk['doc_name']}\nContent: {chunk['content']}")
            relevant_chunks.append({
                "content": chunk["content"],
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
                "similarity": similarity,
                "chunk_id": chunk["id"]
            })
        
        context = "\n\n".join(context_chunks)
        
        # Generate answer using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based on the provided document content. Always provide complete and detailed answers without using markdown formatting. Do not use asterisks, bold, or any special formatting. Provide clean, plain text answers. Include specific references to the documents when possible."
                },
                {
                    "role": "user", 
                    "content": f"Document content:\n{context}\n\nQuestion: {query}\n\nPlease provide a complete and detailed answer in plain text without any formatting:"
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        answer = clean_markdown(answer)
        
        # Prepare citations
        citations = []
        for result in relevant_chunks[:3]:  # Top 3 citations
            citations.append({
                "text": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                "page": result["page"],
                "source": result["doc_name"],
                "similarity": round(result["similarity"], 3),
                "chunk_id": result["chunk_id"]
            })
        
        processing_time = time.time() - start_time
        
        # Update analytics
        analytics["query_times"].append(processing_time)
        if len(analytics["query_times"]) > 100:  # Keep only last 100 queries
            analytics["query_times"] = analytics["query_times"][-100:]
        
        result = {
            "answer": answer,
            "citations": citations,
            "sources": [{"page": result["page"], "source": result["doc_name"]} for result in relevant_chunks[:3]],
            "processing_time": round(processing_time, 2),
            "chunks_used": len(relevant_chunks),
            "cached": False
        }
        
        # Cache the result
        query_cache[query_hash] = result
        if len(query_cache) > 1000:  # Limit cache size
            # Remove oldest entries
            oldest_keys = list(query_cache.keys())[:100]
            for key in oldest_keys:
                del query_cache[key]
        
        save_data()
        
        return result
        
    except Exception as e:
        analytics["error_counts"]["qa_processing"] += 1
        save_data()
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/api/session")
async def get_session_info(request: Request, response: Response):
    """Get or create session information"""
    session_id = get_session_id(request)
    
    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax"
    )
    
    # Count documents for this session
    document_count = len(documents.get(session_id, {}))
    
    return {
        "session_id": session_id,
        "created_at": sessions[session_id]["created_at"],
        "last_activity": sessions[session_id]["last_activity"],
        "document_count": document_count
    }

@app.delete("/api/session")
async def clear_session(request: Request, response: Response):
    """Clear current session"""
    session_id = get_session_id(request)
    
    # Remove session data
    if session_id in documents:
        del documents[session_id]
    if session_id in sessions:
        del sessions[session_id]
    
    # Remove session cookie
    response.delete_cookie("session_id")
    
    save_data()
    
    return {"message": "Session cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
