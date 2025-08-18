from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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
import tiktoken
import numpy as np
import time

# Set OpenAI API key
openai.api_key = "your_openai_api_key_here"

app = FastAPI(title="Working RAG Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
documents = {}
chunks = []
sessions = {}
DATA_FILE = "data_working.pkl"

class RAGProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into chunks with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

rag_processor = RAGProcessor()

def save_data():
    """Save data to disk"""
    data = {
        "documents": documents,
        "chunks": chunks,
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
                global documents, chunks, sessions
                documents = data.get("documents", {})
                chunks = data.get("chunks", [])
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

def get_session_id(request: Request) -> str:
    """Get or create session ID from cookies"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "user_id": "demo_user",
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        save_data()
    else:
        sessions[session_id]["last_activity"] = datetime.now()
    return session_id

def process_document_sync(file_id: str, file_path: str, filename: str, file_ext: str):
    """Synchronous document processing"""
    try:
        # Update status to extracting
        documents[file_id] = {
            "id": file_id,
            "filename": filename,
            "status": "extracting",
            "created_at": datetime.now().isoformat(),
        }
        save_data()
        
        # Extract text
        text_content = extract_text_from_file(file_path, file_ext)
        
        if not text_content.strip():
            documents[file_id]["status"] = "failed"
            documents[file_id]["error"] = "No text content found"
            save_data()
            return False
        
        # Update status to chunking
        documents[file_id]["status"] = "chunking"
        save_data()
        
        # Chunk the content
        text_chunks = rag_processor.chunk_text(text_content)
        
        # Process each chunk
        chunk_records = []
        for i, chunk in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            
            # Get embedding
            embedding = rag_processor.get_embedding(chunk)
            
            # Create chunk record with metadata
            chunk_record = {
                "id": chunk_id,
                "doc_id": file_id,
                "doc_name": filename,
                "content": chunk,
                "embedding": embedding,
                "chunk_index": i,
                "page": 1,  # For now, assume single page
                "created_at": datetime.now().isoformat()
            }
            
            chunk_records.append(chunk_record)
            chunks.append(chunk_record)
        
        # Update document status
        documents[file_id]["status"] = "indexed"
        documents[file_id]["chunk_count"] = len(text_chunks)
        documents[file_id]["content"] = text_content
        save_data()
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return True
            
    except Exception as e:
        documents[file_id]["status"] = "failed"
        documents[file_id]["error"] = str(e)
        save_data()
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return False

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Final RAG API is running",
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "sessions_count": len(sessions)
    }

@app.get("/api/debug")
async def debug_info():
    """Debug endpoint to see what's in memory"""
    return {
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "sessions_count": len(sessions),
        "document_ids": list(documents.keys()),
        "sample_document": list(documents.values())[0] if documents else None
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported. Please upload PDF, DOCX, or TXT files.")
        
        # Validate file size (30MB limit)
        if file.size > 30 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum size is 30MB.")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save file temporarily
        file_path = f"temp_{file_id}{file_ext}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record
        doc = {
            "id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "created_at": datetime.now().isoformat(),
            "content": ""
        }
        documents[file_id] = doc
        save_data()
        
        # Process document synchronously
        success = process_document_sync(file_id, file_path, file.filename, file_ext)
        
        if success:
            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "indexed",
                "message": "File uploaded and processed successfully."
            }
        else:
            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "failed",
                "message": "File processing failed."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    """Get all documents for the current session"""
    try:
        # Debug prints
        print(f"DEBUG: documents dict has {len(documents)} items")
        print(f"DEBUG: documents keys: {list(documents.keys())}")
        
        # Return all documents without session filtering
        docs = []
        for doc_id, doc in documents.items():
            docs.append({
                "id": doc_id,
                "filename": doc["filename"],
                "status": doc["status"],
                "created_at": doc["created_at"],
                "chunk_count": len([c for c in chunks if c.get("doc_id") == doc_id]),
                "content": doc.get("content", "")
            })
        
        print(f"DEBUG: returning {len(docs)} documents")
        return docs
    except Exception as e:
        print(f"DEBUG: Error in get_documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.get("/api/documents/{doc_id}/status")
async def get_document_status(doc_id: str):
    """Get status of a specific document"""
    try:
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = documents[doc_id]
        return {
            "id": doc_id,
            "filename": doc["filename"],
            "status": doc["status"],
            "created_at": doc["created_at"],
            "chunk_count": len([c for c in chunks if c.get("doc_id") == doc_id]),
            "error": doc.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")

@app.post("/api/qa")
async def ask_question(request: Request):
    """Ask a question about documents"""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        document_id = data.get("document_id")
        scope = data.get("scope", "all")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Get all documents or specific document
        if scope == "all" or not document_id:
            available_docs = [doc for doc in documents.values() if doc["status"] == "indexed"]
            if not available_docs:
                return {
                    "answer": "No documents uploaded yet. Please upload a document first.",
                    "citations": [],
                    "sources": [],
                    "processing_time": 0,
                    "chunks_used": 0
                }
        else:
            if document_id not in documents:
                raise HTTPException(status_code=404, detail="Document not found")
            if documents[document_id]["status"] != "indexed":
                raise HTTPException(status_code=400, detail="Document is not ready for querying")
        
        start_time = time.time()
        
        # Get relevant chunks
        if scope == "all" or not document_id:
            relevant_chunks = rag_processor.search(question, top_k=8)
        else:
            relevant_chunks = rag_processor.search(question, doc_id=document_id, top_k=8)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "citations": [],
                "sources": [],
                "processing_time": time.time() - start_time,
                "chunks_used": 0
            }
        
        # Prepare context for LLM
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        
        # Generate answer using OpenAI
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided document context. Always provide accurate, detailed answers based on the information given. If the context doesn't contain enough information to answer the question, say so. Do not use markdown formatting in your response."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a comprehensive answer based on the context above. If the context doesn't contain enough information, say so."
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            answer = clean_markdown(answer)
            
            # Prepare citations
            citations = []
            sources = []
            for i, chunk in enumerate(relevant_chunks):
                citations.append({
                    "text": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "page": chunk.get("page", 1),
                    "source": chunk.get("filename", "Unknown"),
                    "similarity": chunk.get("similarity", 0),
                    "chunk_id": chunk["id"]
                })
                sources.append({
                    "page": chunk.get("page", 1),
                    "source": chunk.get("filename", "Unknown")
                })
            
            return {
                "answer": answer,
                "citations": citations,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "chunks_used": len(relevant_chunks)
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations": [],
                "sources": [],
                "processing_time": time.time() - start_time,
                "chunks_used": 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/api/documents/{doc_id}/content")
async def get_document_content(doc_id: str):
    """Get content of a specific document"""
    try:
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = documents[doc_id]
        doc_chunks = [c for c in chunks if c.get("doc_id") == doc_id]
        
        return {
            "id": doc_id,
            "filename": doc["filename"],
            "content": doc.get("content", ""),
            "chunks": [
                {
                    "id": c["id"],
                    "content": c["content"],
                    "chunk_index": c["chunk_index"],
                    "page": c.get("page", 1)
                }
                for c in doc_chunks
            ],
            "status": doc["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {str(e)}")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document"""
    try:
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove document
        del documents[doc_id]
        
        # Remove associated chunks
        global chunks
        chunks = [c for c in chunks if c.get("doc_id") != doc_id]
        
        save_data()
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/api/session")
async def get_session_info(request: Request):
    """Get session information"""
    session_id = get_session_id(request)
    session = sessions.get(session_id)
    if session:
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat()
        }
    return {"session_id": session_id, "status": "new"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
