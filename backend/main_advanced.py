from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import openai
import PyPDF2
import re
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from docx import Document as DocxDocument
import numpy as np
import time
from PIL import Image
import pytesseract
import io
import logging
from collections import defaultdict, Counter
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Advanced RAG Document Q&A API")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents = {}
chunks = []
sessions = {}
DATA_FILE = "data_advanced.pkl"

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
    session_id = request.cookies.get("session_id")
    
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    
    sessions[session_id]["last_activity"] = datetime.now().isoformat()
    analytics["session_activity"][session_id] += 1
    return session_id

def cleanup_old_sessions():
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
    
    global chunks
    chunks = [c for c in chunks if c.get("session_id") not in expired_sessions]

def save_data():
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

load_data()

class RAGProcessor:
    def __init__(self):
        self.chunk_size = 800
        self.chunk_overlap = 150
    
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in embedding_cache:
            return embedding_cache[text_hash]
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            
            embedding_cache[text_hash] = embedding
            analytics["api_usage"]["embeddings_generated"] += 1
            analytics["api_usage"]["tokens_processed"] += len(text.split())
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

rag_processor = RAGProcessor()

def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text.strip()

def extract_text_from_file(file_path: str, file_type: str) -> str:
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
    try:
        image = Image.open(image_path)
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        except Exception as e:
            print(f"Tesseract failed: {e}")
            return "OCR failed to extract text from this image."
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def process_document_sync(file_id: str, file_path: str, filename: str, file_ext: str, session_id: str):
    try:
        documents[session_id][file_id]["status"] = "extracting"
        save_data()
        
        text_content = extract_text_from_file(file_path, file_ext)
        
        documents[session_id][file_id]["status"] = "chunking"
        save_data()
        
        text_chunks = rag_processor.chunk_text(text_content)
        
        for i, chunk in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            
            embedding = rag_processor.get_embedding(chunk)
            
            chunk_record = {
                "id": chunk_id,
                "doc_id": file_id,
                "doc_name": filename,
                "session_id": session_id,
                "content": chunk,
                "embedding": embedding,
                "chunk_index": i,
                "page": 1,
                "created_at": datetime.now().isoformat()
            }
            
            chunks.append(chunk_record)
        
        documents[session_id][file_id]["status"] = "indexed"
        documents[session_id][file_id]["chunk_count"] = len(text_chunks)
        documents[session_id][file_id]["content"] = text_content
        documents[session_id][file_id]["file_type"] = file_ext
        
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
    cleanup_old_sessions()
    
    avg_query_time = np.mean(analytics["query_times"]) if analytics["query_times"] else 0
    total_sessions = len(sessions)
    active_sessions = len([s for s in sessions.values() 
                          if datetime.fromisoformat(s["last_activity"]) > datetime.now() - timedelta(hours=1)])
    
    return {
        "status": "healthy",
        "message": "Advanced RAG API is running",
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
    session_id = get_session_id(request)
    
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
    session_id = get_session_id(request)
    
    if session_id not in documents:
        documents[session_id] = {}
    
    allowed_extensions = {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported")
    
    max_size = 30 * 1024 * 1024
    if file.size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 30MB")
    
    file_id = str(uuid.uuid4())
    
    file_path = f"temp_{file_id}{file_ext}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
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
        
        logger.info(f"Processing document {file_id} for session {session_id}")
        success = process_document_sync(file_id, file_path, file.filename, file_ext, session_id)
        
        if success:
            logger.info(f"Document {file_id} processed successfully")
            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "indexed",
                "message": "File uploaded and processed successfully."
            }
        else:
            logger.error(f"Document {file_id} processing failed")
            raise HTTPException(status_code=500, detail="Failed to process document")
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents(request: Request):
    session_id = get_session_id(request)
    
    user_docs = documents.get(session_id, {})
    
    logger.info(f"Session ID: {session_id}")
    logger.info(f"User docs count: {len(user_docs)}")
    logger.info(f"All documents: {documents}")
    
    if len(user_docs) == 0 and len(documents) > 0:
        recent_sessions = []
        for sess_id, docs in documents.items():
            if docs:
                latest_time = max(doc.get("created_at", "") for doc in docs.values())
                recent_sessions.append((sess_id, latest_time))
        
        if recent_sessions:
            recent_sessions.sort(key=lambda x: x[1], reverse=True)
            most_recent_session = recent_sessions[0][0]
            user_docs = documents.get(most_recent_session, {})
            logger.info(f"Using documents from recent session: {most_recent_session}")
    
    shared_docs = []
    for doc_id, share_info in shared_documents.items():
        if session_id in share_info.get("shared_with", []):
            for owner_session, docs in documents.items():
                if doc_id in docs:
                    doc = docs[doc_id].copy()
                    doc["shared_by"] = share_info["shared_by"]
                    doc["permissions"] = share_info["permissions"]
                    shared_docs.append(doc)
                    break
    
    all_docs = []
    for doc in user_docs.values():
        doc_copy = doc.copy()
        doc_copy["ownership"] = "own"
        all_docs.append(doc_copy)
    
    for doc in shared_docs:
        doc_copy = doc.copy()
        doc_copy["ownership"] = "shared"
        all_docs.append(doc_copy)
    
    logger.info(f"Returning {len(all_docs)} documents")
    return all_docs

@app.post("/api/documents/{doc_id}/share")
async def share_document(doc_id: str, request: Request):
    session_id = get_session_id(request)
    
    if session_id not in documents or doc_id not in documents[session_id]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    data = await request.json()
    target_session = data.get("session_id")
    permissions = data.get("permissions", "read")
    
    if not target_session:
        raise HTTPException(status_code=400, detail="Target session ID required")
    
    if doc_id not in shared_documents:
        shared_documents[doc_id] = {
            "shared_by": session_id,
            "shared_with": [],
            "permissions": "read"
        }
    
    if target_session not in shared_documents[doc_id]["shared_with"]:
        shared_documents[doc_id]["shared_with"].append(target_session)
    
    save_data()
    
    return {"message": f"Document shared with session {target_session}"}

@app.post("/api/documents/{doc_id}/comment")
async def add_comment(doc_id: str, request: Request):
    session_id = get_session_id(request)
    
    data = await request.json()
    comment_text = data.get("comment")
    
    if not comment_text:
        raise HTTPException(status_code=400, detail="Comment text required")
    
    if doc_id not in document_comments:
        document_comments[doc_id] = []
    
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
    session_id = get_session_id(request)
    
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
    session_id = get_session_id(request)
    
    doc = None
    if session_id in documents and doc_id in documents[session_id]:
        doc = documents[session_id][doc_id]
    elif doc_id in shared_documents and session_id in shared_documents[doc_id]["shared_with"]:
        for owner_session, docs in documents.items():
            if doc_id in docs:
                doc = docs[doc_id]
                break
    
    if not doc:
        for owner_session, docs in documents.items():
            if doc_id in docs:
                doc = docs[doc_id]
                logger.info(f"Found document {doc_id} in session {owner_session} (fallback)")
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
    session_id = get_session_id(request)
    
    if session_id not in documents or doc_id not in documents[session_id]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents[session_id][doc_id]
    return doc

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    session_id = get_session_id(request)
    
    logger.info(f"Attempting to delete document {doc_id} from session {session_id}")
    logger.info(f"Available documents: {list(documents.keys())}")
    
    doc_found = False
    for owner_session, docs in documents.items():
        logger.info(f"Checking session {owner_session} with {len(docs)} documents")
        if doc_id in docs:
            del documents[owner_session][doc_id]
            doc_found = True
            logger.info(f"Successfully deleted document {doc_id} from session {owner_session}")
            break
    
    if not doc_found:
        logger.error(f"Document {doc_id} not found in any session")
        raise HTTPException(status_code=404, detail="Document not found")
    
    global chunks
    chunks = [c for c in chunks if c["doc_id"] != doc_id]
    
    if doc_id in shared_documents:
        del shared_documents[doc_id]
    
    if doc_id in document_comments:
        del document_comments[doc_id]
    
    save_data()
    
    return {"message": "Document deleted successfully"}

@app.post("/api/qa")
async def ask_question(request: Request):
    session_id = get_session_id(request)
    
    data = await request.json()
    query = data.get("question", "")
    doc_id = data.get("document_id")
    scope = data.get("scope", "all")
    
    if not query:
        raise HTTPException(status_code=400, detail="Question is required")
    
    analytics["total_queries"] += 1
    analytics["popular_queries"][query.lower()[:50]] += 1
    
    query_hash = hashlib.md5(f"{query}_{doc_id}_{scope}".encode()).hexdigest()
    if query_hash in query_cache:
        cached_result = query_cache[query_hash]
        cached_result["cached"] = True
        return cached_result
    
    session_docs = []
    for owner_session, docs in documents.items():
        for doc in docs.values():
            if doc.get("status") == "indexed":
                session_docs.append(doc)
    
    logger.info(f"Found {len(session_docs)} total indexed documents across all sessions")
    
    if not session_docs:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "citations": [],
            "sources": []
        }
    
    try:
        start_time = time.time()
        
        query_embedding = rag_processor.get_embedding(query)
        if not query_embedding:
            return {
                "answer": "Error: Could not process query embedding.",
                "citations": [],
                "sources": []
            }
        
        candidate_chunks = [c for c in chunks]
        logger.info(f"Found {len(candidate_chunks)} total chunks across all sessions")
        
        if scope == "document" and doc_id:
            candidate_chunks = [chunk for chunk in candidate_chunks if chunk["doc_id"] == doc_id]
        
        similarities = []
        for chunk in candidate_chunks:
            if chunk.get("embedding"):
                similarity = rag_processor.cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append((similarity, chunk))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:8]
        
        if not top_results:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "citations": [],
                "sources": []
            }
        
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
        
        citations = []
        for result in relevant_chunks[:3]:
            citations.append({
                "text": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                "page": result["page"],
                "source": result["doc_name"],
                "similarity": round(result["similarity"], 3),
                "chunk_id": result["chunk_id"]
            })
        
        processing_time = time.time() - start_time
        
        analytics["query_times"].append(processing_time)
        if len(analytics["query_times"]) > 100:
            analytics["query_times"] = analytics["query_times"][-100:]
        
        result = {
            "answer": answer,
            "citations": citations,
            "sources": [{"page": result["page"], "source": result["doc_name"]} for result in relevant_chunks[:3]],
            "processing_time": round(processing_time, 2),
            "chunks_used": len(relevant_chunks),
            "cached": False
        }
        
        query_cache[query_hash] = result
        if len(query_cache) > 1000:
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
    session_id = get_session_id(request)
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=86400,
        samesite="lax"
    )
    
    document_count = len(documents.get(session_id, {}))
    
    return {
        "session_id": session_id,
        "created_at": sessions[session_id]["created_at"],
        "last_activity": sessions[session_id]["last_activity"],
        "document_count": document_count
    }

@app.delete("/api/session")
async def clear_session(request: Request, response: Response):
    session_id = get_session_id(request)
    
    if session_id in documents:
        del documents[session_id]
    if session_id in sessions:
        del sessions[session_id]
    
    response.delete_cookie("session_id")
    
    save_data()
    
    return {"message": "Session cleared successfully"}

@app.get("/api/chunks")
async def get_chunks(request: Request):
    all_chunks = []
    
    for chunk in chunks:
        chunk_info = {
            "id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "doc_name": chunk["doc_name"],
            "content": chunk["content"],
            "chunk_index": chunk.get("chunk_index", 0),
            "page": chunk.get("page", 1),
            "created_at": chunk.get("created_at", ""),
            "session_id": chunk.get("session_id", "")
        }
        all_chunks.append(chunk_info)
    
    return {
        "total_chunks": len(all_chunks),
        "chunks": all_chunks
    }

@app.get("/api/chunks/{doc_id}")
async def get_document_chunks(doc_id: str, request: Request):
    document_chunks = []
    
    for chunk in chunks:
        if chunk["doc_id"] == doc_id:
            chunk_info = {
                "id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "doc_name": chunk["doc_name"],
                "content": chunk["content"],
                "chunk_index": chunk.get("chunk_index", 0),
                "page": chunk.get("page", 1),
                "created_at": chunk.get("created_at", ""),
                "session_id": chunk.get("session_id", "")
            }
            document_chunks.append(chunk_info)
    
    if not document_chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this document")
    
    return {
        "doc_id": doc_id,
        "total_chunks": len(document_chunks),
        "chunks": document_chunks
    }

@app.get("/api/chunking-stats")
async def get_chunking_stats(request: Request):
    doc_chunks = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(chunk)
    
    stats = []
    for doc_id, doc_chunk_list in doc_chunks.items():
        doc_name = doc_chunk_list[0]["doc_name"] if doc_chunk_list else "Unknown"
        
        total_length = sum(len(chunk["content"]) for chunk in doc_chunk_list)
        avg_length = total_length / len(doc_chunk_list) if doc_chunk_list else 0
        
        stats.append({
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_count": len(doc_chunk_list),
            "avg_chunk_length": round(avg_length, 2),
            "total_content_length": total_length
        })
    
    return {
        "total_documents": len(stats),
        "total_chunks": len(chunks),
        "average_chunks_per_doc": round(len(chunks) / len(stats), 2) if stats else 0,
        "document_stats": stats
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    port_str = os.getenv("PORT", "8001")
    
    if port_str == "$PORT" or not port_str.isdigit():
        port = 8001
    else:
        port = int(port_str)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
