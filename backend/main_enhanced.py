from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
from rag_service import RAGService
import asyncio
import time

# Set OpenAI API key
openai.api_key = "your_openai_api_key_here"

app = FastAPI(title="Enhanced RAG Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

# Simple session management (in-memory for demo)
sessions = {}
session_cookies = {}

# Data persistence
DATA_FILE = "data.pkl"

class SessionManager:
    @staticmethod
    def create_session(user_id: str = "demo_user") -> str:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        if session_id in sessions:
            sessions[session_id]["last_activity"] = datetime.now()
            return sessions[session_id]
        return None
    
    @staticmethod
    def cleanup_old_sessions():
        """Clean up sessions older than 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        expired = [sid for sid, session in sessions.items() 
                  if session["last_activity"] < cutoff]
        for sid in expired:
            del sessions[sid]

def get_session_id(request: Request) -> str:
    """Get or create session ID from cookies"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        session_id = SessionManager.create_session()
    return session_id

def save_data():
    """Save data to disk"""
    data = {
        "documents": rag_service.documents,
        "chunks": rag_service.chunks,
        "sessions": sessions
    }
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_data():
    """Load data from disk"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                rag_service.documents = data.get("documents", {})
                rag_service.chunks = data.get("chunks", [])
                global sessions
                sessions = data.get("sessions", {})
        except Exception as e:
            print(f"Error loading data: {e}")

# Load existing data on startup
load_data()

def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text"""
    # Remove bold formatting (**text** -> text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove italic formatting (*text* -> text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove code formatting (`text` -> text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove headers (# text -> text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text from different file types"""
    try:
        if file_type == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()

        elif file_type == '.docx':
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()

        elif file_type == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        raise Exception(f"Error extracting text from {file_type} file: {str(e)}")

async def process_document_background(file_id: str, file_path: str, filename: str, file_ext: str, session_id: str):
    """Background task for document processing with status updates"""
    try:
        # Update status to extracting
        rag_service.documents[file_id] = {
            "id": file_id,
            "filename": filename,
            "status": "extracting",
            "created_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        save_data()
        
        # Extract text
        text_content = extract_text_from_file(file_path, file_ext)
        
        if not text_content.strip():
            rag_service.documents[file_id]["status"] = "failed"
            rag_service.documents[file_id]["error"] = "No text content found"
            save_data()
            return
        
        # Update status to chunking
        rag_service.documents[file_id]["status"] = "chunking"
        save_data()
        
        # Process with RAG service
        result = rag_service.process_document(file_id, filename, text_content)
        
        if result["success"]:
            rag_service.documents[file_id]["status"] = "indexed"
            rag_service.documents[file_id]["chunk_count"] = result["chunks_created"]
        else:
            rag_service.documents[file_id]["status"] = "failed"
            rag_service.documents[file_id]["error"] = result["error"]
        
        save_data()
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        rag_service.documents[file_id]["status"] = "failed"
        rag_service.documents[file_id]["error"] = str(e)
        save_data()
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Enhanced RAG API is running",
        "documents_count": len(rag_service.documents),
        "chunks_count": len(rag_service.chunks),
        "sessions_count": len(sessions)
    }

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file type
    allowed_types = ['.pdf', '.docx', '.txt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported types: {', '.join(allowed_types)}")

    # Check file size (30MB limit)
    max_size = 30 * 1024 * 1024
    if file.size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 30MB")

    try:
        # Save file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_ext}"
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get or create session
        session_id = get_session_id(request) if request else SessionManager.create_session()

        # Create initial document record
        doc = {
            "id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "created_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        rag_service.documents[file_id] = doc
        save_data()

        # Start background processing
        if background_tasks:
            background_tasks.add_task(
                process_document_background, 
                file_id, file_path, file.filename, file_ext, session_id
            )

        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "File uploaded successfully. Processing in background."
        }

    except Exception as e:
        # Clean up file if it was created
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents(request: Request):
    """Get documents for current session"""
    session_id = get_session_id(request)
    # Filter documents by session
    session_docs = [
        doc for doc in rag_service.documents.values() 
        if doc.get("session_id") == session_id
    ]
    return session_docs

@app.get("/api/documents/{doc_id}/status")
async def get_document_status(doc_id: str, request: Request):
    """Get specific document status"""
    session_id = get_session_id(request)
    doc = rag_service.documents.get(doc_id)
    if not doc or doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.post("/api/qa")
async def ask_question(request: dict, req: Request):
    query = request.get("question", "")
    doc_id = request.get("document_id")
    scope = request.get("scope", "all")
    
    if not query:
        raise HTTPException(status_code=400, detail="Question is required")

    session_id = get_session_id(req)

    # Filter documents by session
    session_docs = [
        doc for doc in rag_service.documents.values() 
        if doc.get("session_id") == session_id and doc.get("status") == "indexed"
    ]
    
    if not session_docs:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "citations": [],
            "sources": []
        }

    try:
        start_time = time.time()
        
        # Search for relevant chunks
        search_doc_id = None
        if scope == "document" and doc_id:
            search_doc_id = doc_id
        
        search_results = rag_service.search(query, search_doc_id, top_k=8)
        
        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "citations": [],
                "sources": []
            }

        # Prepare context for LLM
        context_chunks = []
        for result in search_results:
            context_chunks.append(f"Document: {result['doc_name']}\nPage: {result['page']}\nContent: {result['content']}")

        context = "\n\n".join(context_chunks)

        # Generate answer using OpenAI
        response = openai.ChatCompletion.create(
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
        for result in search_results[:3]:  # Top 3 citations
            citations.append({
                "text": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                "page": result["page"],
                "source": result["doc_name"],
                "similarity": round(result["similarity"], 3),
                "chunk_id": result["chunk_id"]
            })

        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "citations": citations,
            "sources": [{"page": result["page"], "source": result["doc_name"]} for result in search_results[:3]],
            "processing_time": round(processing_time, 2),
            "chunks_used": len(search_results)
        }

    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "citations": [],
            "sources": []
        }

@app.get("/api/documents/{doc_id}/content")
async def get_document_content(doc_id: str, request: Request):
    """Get document content for viewer"""
    session_id = get_session_id(request)
    doc = rag_service.documents.get(doc_id)
    if not doc or doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get chunks for this document
    doc_chunks = [chunk for chunk in rag_service.chunks if chunk["doc_id"] == doc_id]
    
    return {
        "id": doc_id,
        "filename": doc["filename"],
        "content": doc.get("content", ""),
        "chunks": doc_chunks,
        "status": doc["status"]
    }

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    """Delete document"""
    session_id = get_session_id(request)
    doc = rag_service.documents.get(doc_id)
    if not doc or doc.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Document not found")
    
    success = rag_service.delete_document(doc_id)
    if success:
        save_data()
        return {"message": "Document deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/api/session")
async def get_session_info(request: Request):
    """Get session information"""
    session_id = get_session_id(request)
    session = SessionManager.get_session(session_id)
    if session:
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat()
        }
    return {"session_id": session_id, "status": "new"}

# Cleanup old sessions periodically
@app.on_event("startup")
async def startup_event():
    SessionManager.cleanup_old_sessions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
