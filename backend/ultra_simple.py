from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from openai import OpenAI
import PyPDF2
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from docx import Document as DocxDocument
import numpy as np
import time
import io

# Initialize FastAPI
app = FastAPI(title="Simple RAG Document Q&A API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "https://rag-document-qa-iota.vercel.app",
        "https://rag-document-qa.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("WARNING: No OpenAI API key found!")
client = OpenAI(api_key=api_key) if api_key else None

# Simple in-memory storage
documents = {}
chunks = []
sessions = {}

def get_session_id(request: Request) -> str:
    """Get or create session ID"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
        }
    return session_id

def simple_chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Simple text chunking"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded file"""
    try:
        if filename.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif filename.lower().endswith('.docx'):
            doc = DocxDocument(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif filename.lower().endswith('.txt'):
            return file_content.decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding"""
    if not client:
        # Return dummy embedding for testing
        return [0.1] * 1536
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.1] * 1536

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity"""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)

def search_chunks(query: str, session_id: str, top_k: int = 5) -> List[Dict]:
    """Search for relevant chunks"""
    query_embedding = get_embedding(query)
    
    # Filter chunks by session
    session_chunks = [c for c in chunks if c.get("session_id") == session_id]
    
    if not session_chunks:
        return []
    
    # Calculate similarities
    scored_chunks = []
    for chunk in session_chunks:
        similarity = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append({
            **chunk,
            "similarity": similarity
        })
    
    # Sort by similarity and return top_k
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]

@app.get("/")
async def root():
    return {"message": "Simple RAG Document Q&A Backend is running", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Simple RAG API is running",
        "api_key": {"status": "configured" if client else "missing"},
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "sessions_count": len(sessions)
    }

@app.get("/api/session")
async def get_session_info(request: Request, response: Response):
    session_id = get_session_id(request)
    response.set_cookie("session_id", session_id, httponly=True)
    return {
        "session_id": session_id,
        "session": sessions[session_id]
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):
    session_id = get_session_id(request)
    
    # Read file content
    content = await file.read()
    
    # Extract text
    text = extract_text_from_file(content, file.filename)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in document")
    
    # Create document ID
    doc_id = str(uuid.uuid4())
    
    # Store document
    documents[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "text": text,
        "session_id": session_id,
        "uploaded_at": datetime.now().isoformat()
    }
    
    # Create chunks
    text_chunks = simple_chunk_text(text)
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc_id}_{i}"
        embedding = get_embedding(chunk_text)
        
        chunks.append({
            "id": chunk_id,
            "doc_id": doc_id,
            "doc_name": file.filename,
            "text": chunk_text,
            "embedding": embedding,
            "session_id": session_id,
            "chunk_index": i
        })
    
    return {
        "message": "File uploaded successfully",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(text_chunks)
    }

@app.get("/api/documents")
async def get_documents(request: Request):
    session_id = get_session_id(request)
    session_docs = [
        {
            "id": doc["id"],
            "filename": doc["filename"],
            "uploaded_at": doc["uploaded_at"]
        }
        for doc in documents.values()
        if doc["session_id"] == session_id
    ]
    return session_docs

@app.get("/api/documents/{doc_id}/content")
async def get_document_content(doc_id: str, request: Request):
    session_id = get_session_id(request)
    
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents[doc_id]
    if doc["session_id"] != session_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get chunks for this document
    doc_chunks = [c for c in chunks if c["doc_id"] == doc_id]
    
    return {
        "document": {
            "id": doc["id"],
            "filename": doc["filename"],
            "text": doc["text"],
            "uploaded_at": doc["uploaded_at"]
        },
        "chunks": [
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in doc_chunks
        ]
    }

@app.post("/api/qa")
async def answer_question(
    request_data: dict,
    request: Request
):
    session_id = get_session_id(request)
    
    question = request_data.get("question", "")
    scope = request_data.get("scope", "all")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not client:
        return {
            "answer": "OpenAI API key not configured. Please add your API key to use the Q&A feature.",
            "sources": [],
            "scope": scope
        }
    
    # Search for relevant chunks
    relevant_chunks = search_chunks(question, session_id, top_k=5)
    
    if not relevant_chunks:
        return {
            "answer": "I couldn't find any relevant information in your documents to answer this question.",
            "sources": [],
            "scope": scope
        }
    
    # Prepare context from chunks
    context = "\n\n".join([
        f"Document: {chunk['doc_name']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    # Generate answer using OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided documents. Use only the information from the documents to answer questions. If the information is not in the documents, say so."
                },
                {
                    "role": "user",
                    "content": f"Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer based on the provided context:"
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        # Prepare sources
        sources = [
            {
                "doc_name": chunk["doc_name"],
                "similarity": round(chunk["similarity"], 3),
                "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            }
            for chunk in relevant_chunks[:3]
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "scope": scope
        }
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {
            "answer": f"Sorry, I encountered an error while generating the answer: {str(e)}",
            "sources": [],
            "scope": scope
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
