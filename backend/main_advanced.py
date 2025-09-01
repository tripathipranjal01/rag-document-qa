from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
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
from PIL import Image
import pytesseract
import easyocr
import io
import logging
from collections import defaultdict, Counter
import hashlib
import re
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it with your actual OpenAI API key.")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://your-frontend-domain.onrender.com").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

app = FastAPI(
    title="Document Q&A API",
    description="AI-powered document analysis and Q&A system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize('NFKD', filename)
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    filename = filename.strip('-_')
    
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:100-len(ext)] + ext
    
    return filename

def validate_file_content(file_content: bytes, file_ext: str) -> bool:
    try:
        content_str = file_content.decode('utf-8', errors='ignore').lower()
        
        dangerous_patterns = [
            '<script', 'javascript:', 'vbscript:', 'data:text/html',
            '<?php', '<?=', '<? ', 'eval(', 'exec(', 'system(',
            'document.cookie', 'window.location', 'alert('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in content_str:
                logger.warning(f"Potentially malicious content detected: {pattern}")
                return False
        
        if file_ext == '.pdf':
            if not file_content.startswith(b'%PDF'):
                return False
        elif file_ext == '.docx':
            if not file_content.startswith(b'PK'):
                return False
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            image_signatures = {
                b'\x89PNG\r\n\x1a\n': '.png',
                b'\xff\xd8\xff': '.jpg',
                b'BM': '.bmp',
                b'II*\x00': '.tiff',
                b'MM\x00*': '.tiff'
            }
            
            is_valid_image = False
            for signature, ext in image_signatures.items():
                if file_content.startswith(signature) and ext == file_ext:
                    is_valid_image = True
                    break
            
            if not is_valid_image:
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating file content: {e}")
        return False

def calculate_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

documents = {}
chunks = {}
sessions = {}
analytics = {
    "total_queries": 0,
    "avg_query_time": 0,
    "processing_times": [],
    "P50": 0,
    "P95": 0,
    "P99": 0
}
chat_history = {}
document_progress = {}

def save_data():
    data = {
        "documents": documents,
        "chunks": chunks,
        "sessions": sessions,
        "analytics": analytics,
        "chat_history": chat_history,
        "document_progress": document_progress
    }
    with open("data_simple.pkl", "wb") as f:
        pickle.dump(data, f)

def load_data():
    global documents, chunks, sessions, analytics, chat_history, document_progress
    try:
        with open("data_simple.pkl", "rb") as f:
            data = pickle.load(f)
            documents = data.get("documents", {})
            chunks = data.get("chunks", {})
            sessions = data.get("sessions", {})
            analytics = data.get("analytics", analytics)
            chat_history = data.get("chat_history", {})
            document_progress = data.get("document_progress", {})
    except FileNotFoundError:
        pass

load_data()

def get_session_id(request: Request) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    
    sessions[session_id]["last_activity"] = datetime.now().isoformat()
    return session_id

def extract_text_from_pdf(pdf_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(docx_content: bytes) -> str:
    try:
        docx_file = io.BytesIO(docx_content)
        doc = DocxDocument(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise HTTPException(status_code=400, detail=f"Error extracting text from DOCX: {str(e)}")

def extract_text_from_image(image_content: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_content))
        
        # Try OCR with pytesseract first
        try:
            text = pytesseract.image_to_string(image)
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"Pytesseract failed: {e}")
        
        # Fallback to EasyOCR
        try:
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image_content)
            text = " ".join([result[1] for result in results])
            return text
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            raise HTTPException(status_code=400, detail="OCR processing failed")
            
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise HTTPException(status_code=400, detail=f"Error extracting text from image: {str(e)}")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        i += chunk_size - overlap
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

def search_similar_chunks(query: str, session_id: str, doc_id: Optional[str] = None, top_k: int = 8) -> List[Dict]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Get relevant chunks
    session_chunks = chunks.get(session_id, [])
    if doc_id:
        session_chunks = [chunk for chunk in session_chunks if chunk.get("document_id") == doc_id]
    
    if not session_chunks:
        return []
    
    # Calculate similarities
    similarities = []
    for chunk in session_chunks:
        similarity = np.dot(query_embedding, chunk["embedding"])
        similarities.append((similarity, chunk))
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in similarities[:top_k]]

def generate_answer(query: str, context_chunks: List[Dict], client: OpenAI) -> str:
    if not context_chunks:
        return "I don't have enough information to answer your question. Please upload some documents first."
    
    context = "\n\n".join([chunk["content"] for chunk in context_chunks])
    
    prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context. 
    If the answer cannot be found in the context, say so. Be concise and accurate.

    Context:
    {context}

    Question: {query}

    Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/")
async def root():
    return {"message": "RAG Document Q&A API is running!"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "analytics": analytics
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):
    session_id = get_session_id(request)
    
    # Validate file size
    max_size = int(os.getenv("MAX_FILE_SIZE", 52428800))  # 50MB default
    if file.size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {max_size} bytes")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed types: {allowed_extensions}")
    
    # Sanitize filename
    if os.getenv("ENABLE_FILENAME_SANITIZATION", "true").lower() == "true":
        original_filename = file.filename
        file.filename = sanitize_filename(file.filename)
        logger.info(f"Sanitized filename: {original_filename} -> {file.filename}")
    
    # Read file content
    file_content = await file.read()
    
    # Validate file content
    if os.getenv("ENABLE_FILE_VALIDATION", "true").lower() == "true":
        if not validate_file_content(file_content, file_ext):
            raise HTTPException(status_code=400, detail="File content validation failed")
    
    # Calculate file hash
    file_hash = calculate_file_hash(file_content)
    
    # Check for duplicate
    for session_docs in documents.values():
        for doc in session_docs.values():
            if doc.get("file_hash") == file_hash:
                raise HTTPException(status_code=400, detail="This file has already been uploaded")
    
    # Initialize progress tracking
    doc_id = str(uuid.uuid4())
    document_progress[doc_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "File uploaded successfully"
    }
    
    try:
        # Extract text based on file type
        if file_ext == '.pdf':
            text = extract_text_from_pdf(file_content)
        elif file_ext == '.docx':
            text = extract_text_from_docx(file_content)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            text = extract_text_from_image(file_content)
        else:
            text = file_content.decode('utf-8', errors='ignore')
        
        document_progress[doc_id]["status"] = "extracting"
        document_progress[doc_id]["progress"] = 25
        document_progress[doc_id]["message"] = "Text extracted successfully"
        
        # Chunk the text
        text_chunks = chunk_text(text)
        
        document_progress[doc_id]["status"] = "chunking"
        document_progress[doc_id]["progress"] = 50
        document_progress[doc_id]["message"] = f"Text chunked into {len(text_chunks)} chunks"
        
        # Get embeddings
        embeddings = get_embeddings(text_chunks)
        
        document_progress[doc_id]["status"] = "embedding"
        document_progress[doc_id]["progress"] = 75
        document_progress[doc_id]["message"] = "Embeddings generated successfully"
        
        # Store document and chunks
        if session_id not in documents:
            documents[session_id] = {}
        
        documents[session_id][doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "file_size": len(file_content),
            "file_type": file_ext,
            "upload_timestamp": datetime.now().isoformat(),
            "processing_status": "completed",
            "total_chunks": len(text_chunks),
            "session_id": session_id,
            "file_hash": file_hash
        }
        
        if session_id not in chunks:
            chunks[session_id] = []
        
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunks[session_id].append({
                "id": chunk_id,
                "document_id": doc_id,
                "chunk_index": i,
                "content": chunk_text,
                "tokens": len(tiktoken.get_encoding("cl100k_base").encode(chunk_text)),
                "embedding": embedding,
                "created_at": datetime.now().isoformat()
            })
        
        document_progress[doc_id]["status"] = "completed"
        document_progress[doc_id]["progress"] = 100
        document_progress[doc_id]["message"] = "Document processed successfully"
        
        save_data()
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": doc_id,
            "filename": file.filename,
            "total_chunks": len(text_chunks),
            "processing_status": "completed"
        }
        
    except Exception as e:
        document_progress[doc_id]["status"] = "failed"
        document_progress[doc_id]["progress"] = 0
        document_progress[doc_id]["message"] = f"Processing failed: {str(e)}"
        
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/api/documents")
async def get_documents(request: Request):
    session_id = get_session_id(request)
    session_docs = documents.get(session_id, {})
    return list(session_docs.values())

@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, request: Request):
    session_id = get_session_id(request)
    
    # Check if user has access to this document
    doc = None
    if session_id in documents and doc_id in documents[session_id]:
        doc = documents[session_id][doc_id]
    else:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get chunks for this document
    session_chunks = chunks.get(session_id, [])
    doc_chunks = [chunk for chunk in session_chunks if chunk.get("document_id") == doc_id]
    
    return {
        "document": doc,
        "chunks": doc_chunks
    }

@app.post("/api/qa")
async def ask_question(request: Request, question_data: dict):
    session_id = get_session_id(request)
    
    question = question_data.get("question", "").strip()
    doc_id = question_data.get("document_id")
    scope = question_data.get("scope", "document")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    start_time = time.time()
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Search for relevant chunks
        if scope == "document" and doc_id:
            context_chunks = search_similar_chunks(question, session_id, doc_id)
        else:
            context_chunks = search_similar_chunks(question, session_id)
        
        # Generate answer
        answer = generate_answer(question, context_chunks, client)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update analytics
        analytics["total_queries"] += 1
        analytics["processing_times"].append(processing_time)
        analytics["avg_query_time"] = sum(analytics["processing_times"]) / len(analytics["processing_times"])
        
        # Calculate percentiles
        if len(analytics["processing_times"]) > 0:
            sorted_times = sorted(analytics["processing_times"])
            analytics["P50"] = sorted_times[len(sorted_times) // 2]
            analytics["P95"] = sorted_times[int(len(sorted_times) * 0.95)]
            analytics["P99"] = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Save chat history
        if session_id not in chat_history:
            chat_history[session_id] = {}
        
        chat_entry = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "document_id": doc_id if scope == "document" else "all",
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "chunk_id": chunk["id"],
                    "text": chunk["content"][:200] + "...",
                    "source": documents[session_id][chunk["document_id"]]["filename"] if chunk["document_id"] in documents[session_id] else "Unknown",
                    "similarity": 0.85
                }
                for chunk in context_chunks[:3]
            ],
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time
        }
        
        # Save to appropriate chat history
        if scope == "document" and doc_id:
            if doc_id not in chat_history[session_id]:
                chat_history[session_id][doc_id] = []
            chat_history[session_id][doc_id].append(chat_entry)
        else:
            if "all" not in chat_history[session_id]:
                chat_history[session_id]["all"] = []
            chat_history[session_id]["all"].append(chat_entry)
        
        save_data()
        
        return {
            "answer": answer,
            "citations": chat_entry["citations"],
            "processing_time": processing_time,
            "chunks_used": len(context_chunks)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/api/qa/stream")
async def ask_question_stream(request: Request, question_data: dict):
    session_id = get_session_id(request)
    
    question = question_data.get("question", "").strip()
    doc_id = question_data.get("document_id")
    scope = question_data.get("scope", "document")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    def generate_stream():
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Search for relevant chunks
            if scope == "document" and doc_id:
                context_chunks = search_similar_chunks(question, session_id, doc_id)
            else:
                context_chunks = search_similar_chunks(question, session_id)
            
            # Stream the answer
            if not context_chunks:
                yield f"data: {json.dumps({'type': 'answer', 'content': 'I don\'t have enough information to answer your question. Please upload some documents first.'})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                return
            
            context = "\n\n".join([chunk["content"] for chunk in context_chunks])
            
            prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context. 
            If the answer cannot be found in the context, say so. Be concise and accurate.

            Context:
            {context}

            Question: {question}

            Answer:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
                stream=True
            )
            
            full_answer = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield f"data: {json.dumps({'type': 'answer', 'content': content})}\n\n"
            
            # Send citations
            citations = [
                {
                    "chunk_id": chunk["id"],
                    "text": chunk["content"][:200] + "...",
                    "source": documents[session_id][chunk["document_id"]]["filename"] if chunk["document_id"] in documents[session_id] else "Unknown",
                    "similarity": 0.85
                }
                for chunk in context_chunks[:3]
            ]
            
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
            # Save chat history
            if session_id not in chat_history:
                chat_history[session_id] = {}
            
            chat_entry = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "document_id": doc_id if scope == "document" else "all",
                "question": question,
                "answer": full_answer,
                "citations": citations,
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0
            }
            
            if scope == "document" and doc_id:
                if doc_id not in chat_history[session_id]:
                    chat_history[session_id][doc_id] = []
                chat_history[session_id][doc_id].append(chat_entry)
            else:
                if "all" not in chat_history[session_id]:
                    chat_history[session_id]["all"] = []
                chat_history[session_id]["all"].append(chat_entry)
            
            save_data()
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing question: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        return StreamingResponse(generate_stream(), media_type="text/plain")

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
    if session_id in chat_history:
        del chat_history[session_id]
    
    response.delete_cookie("session_id")
    
    save_data()
    
    return {"message": "Session cleared successfully"}

@app.get("/api/chat-history/all")
async def get_all_chat_history(request: Request):
    session_id = get_session_id(request)
    
    session_chat_history = chat_history.get(session_id, {})
    all_messages = []
    
    for doc_id_inner, messages in session_chat_history.items():
        all_messages.extend(messages)
    
    all_messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    all_messages = all_messages[:50]
    
    return {"messages": all_messages}

@app.get("/api/chat-history/{doc_id}")
async def get_chat_history(doc_id: str, request: Request):
    session_id = get_session_id(request)
    
    doc = None
    if session_id in documents and doc_id in documents[session_id]:
        doc = documents[session_id][doc_id]
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    session_chat_history = chat_history.get(session_id, {})
    doc_chat_history = session_chat_history.get(doc_id, [])
    
    return {"messages": doc_chat_history}

@app.get("/api/progress/{doc_id}")
async def get_document_progress(doc_id: str):
    progress = document_progress.get(doc_id, {
        "status": "not_found",
        "progress": 0,
        "message": "Document not found"
    })
    return progress

@app.get("/api/security/status")
async def get_security_status():
    return {
        "security_status": "enabled",
        "features": {
            "filename_sanitization": os.getenv("ENABLE_FILENAME_SANITIZATION", "true").lower() == "true",
            "file_validation": os.getenv("ENABLE_FILE_VALIDATION", "true").lower() == "true",
            "file_size_limit": int(os.getenv("MAX_FILE_SIZE", 52428800)),
            "allowed_extensions": ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        }
    }

@app.get("/api/cost/estimate")
async def get_cost_estimate():
    total_tokens = sum(chunk.get("tokens", 0) for session_chunks in chunks.values() for chunk in session_chunks)
    estimated_monthly_cost = (total_tokens / 1000) * 0.00002  # Approximate cost per 1K tokens
    
    return {
        "estimated_monthly_cost": estimated_monthly_cost,
        "total_tokens": total_tokens,
        "free_tier_usage": "OpenAI API has usage-based pricing"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
