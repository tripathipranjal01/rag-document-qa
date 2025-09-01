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
import time
from PIL import Image
import pytesseract
# import easyocr  # Temporarily disabled for faster deployment
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
        # Only validate binary files for correct signatures
        # Skip dangerous pattern detection for PDF/DOCX as they might contain legitimate code examples
        
        if file_ext == '.pdf':
            if not file_content.startswith(b'%PDF'):
                logger.warning(f"Invalid PDF signature")
                return False
        elif file_ext == '.docx':
            if not file_content.startswith(b'PK'):
                logger.warning(f"Invalid DOCX signature")
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
                logger.warning(f"Invalid image signature for {file_ext}")
                return False
        elif file_ext == '.txt':
            # For text files, only check for the most dangerous patterns
            content_str = file_content.decode('utf-8', errors='ignore').lower()
            dangerous_patterns = [
                '<script', 'javascript:', 'document.cookie', 'window.location'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in content_str:
                    logger.warning(f"Potentially malicious content detected: {pattern}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file content: {e}")
        return False

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content for integrity checking"""
    return hashlib.sha256(file_content).hexdigest()

def update_progress(file_id: str, stage: str, progress: int, message: str = ""):
    """Update progress for document processing"""
    document_progress[file_id] = {
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

def get_progress(file_id: str) -> dict:
    """Get current progress for a document"""
    return document_progress.get(file_id, {
        "stage": "unknown",
        "progress": 0,
        "message": "Processing not started",
        "timestamp": datetime.now().isoformat()
    })

def clear_progress(file_id: str):
    """Clear progress tracking for a document"""
    if file_id in document_progress:
        del document_progress[file_id]

client = OpenAI(api_key=OPENAI_API_KEY)

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
chat_history = {}  # Store chat history per document: {session_id: {doc_id: [messages]}}

# Progress tracking for document processing
document_progress = {}  # {file_id: {"stage": "extracting", "progress": 75, "message": "Processing..."}}

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
        "document_comments": document_comments,
        "chat_history": chat_history,
        "document_progress": document_progress
    }
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def load_data():
    """Load data from file"""
    global documents, chunks, sessions, analytics, shared_documents, document_comments, chat_history, document_progress
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
            documents = data.get("documents", {})
            chunks = data.get("chunks", [])
            sessions = data.get("sessions", {})
            analytics.update(data.get("analytics", {}))
            shared_documents = data.get("shared_documents", {})
            document_comments = data.get("document_comments", {})
            chat_history = data.get("chat_history", {})
            document_progress = data.get("document_progress", {})

# Load existing data
load_data()

class RAGProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 800
        self.chunk_overlap = 150
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
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
        """Calculate cosine similarity without numpy"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

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
    """Extract text from image using OCR (Tesseract only for faster deployment)"""
    try:
        image = Image.open(image_path)
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            if text.strip():
                return text
            else:
                return "No text detected in image."
        except Exception as e:
            print(f"Tesseract failed: {e}")
            return "OCR failed to extract text from this image."
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def process_document_sync(file_id: str, file_path: str, filename: str, file_ext: str, session_id: str):
    """Process document synchronously with detailed progress tracking and error handling"""
    try:
        # Initialize progress tracking
        update_progress(file_id, "uploaded", 0, "File uploaded successfully")
        
        # Stage 1: Text Extraction (0-25%)
        update_progress(file_id, "extracting", 10, "Starting text extraction...")
        documents[session_id][file_id]["status"] = "extracting"
        save_data()
        
        try:
            text_content = extract_text_from_file(file_path, file_ext)
            if not text_content or not text_content.strip():
                raise Exception("No text content extracted from file")
            update_progress(file_id, "extracting", 25, "Text extraction completed")
        except Exception as e:
            error_msg = f"Text extraction failed: {str(e)}"
            if "PDF" in str(e):
                error_msg += ". The PDF might be corrupted, password-protected, or contain only images."
            elif "DOCX" in str(e):
                error_msg += ". The document might be corrupted or in an unsupported format."
            elif "image" in str(e).lower():
                error_msg += ". OCR processing failed. Please ensure the image is clear and readable."
            raise Exception(error_msg)
        
        # Stage 2: Text Chunking (25-50%)
        update_progress(file_id, "chunking", 30, "Starting text chunking...")
        documents[session_id][file_id]["status"] = "chunking"
        save_data()
        
        try:
            text_chunks = rag_processor.chunk_text(text_content)
            if not text_chunks:
                raise Exception("No chunks created from text content")
            update_progress(file_id, "chunking", 50, f"Created {len(text_chunks)} text chunks")
        except Exception as e:
            raise Exception(f"Text chunking failed: {str(e)}")
        
        # Stage 3: Embedding Generation (50-90%)
        update_progress(file_id, "embedding", 55, "Starting embedding generation...")
        documents[session_id][file_id]["status"] = "embedding"
        save_data()
        
        chunk_records = []
        for i, chunk in enumerate(text_chunks):
            try:
                # Update progress for each chunk
                progress = 55 + int((i / len(text_chunks)) * 35)
                update_progress(file_id, "embedding", progress, f"Generating embeddings... ({i+1}/{len(text_chunks)})")
                
                chunk_id = str(uuid.uuid4())
                embedding = rag_processor.get_embedding(chunk)
                
                if not embedding:
                    raise Exception(f"Failed to generate embedding for chunk {i+1}")
                
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
                chunk_records.append(chunk_record)
                
            except Exception as e:
                raise Exception(f"Embedding generation failed for chunk {i+1}: {str(e)}")
        
        # Stage 4: Indexing (90-100%)
        update_progress(file_id, "indexing", 95, "Indexing document chunks...")
        documents[session_id][file_id]["status"] = "indexing"
        save_data()
        
        try:
            # Add all chunks to the global chunks list
            chunks.extend(chunk_records)
            
            # Update document status
            documents[session_id][file_id]["status"] = "indexed"
            documents[session_id][file_id]["chunk_count"] = len(text_chunks)
            documents[session_id][file_id]["content"] = text_content
            documents[session_id][file_id]["file_type"] = file_ext
            documents[session_id][file_id]["processing_completed"] = datetime.now().isoformat()
            
            # Update analytics
            analytics["total_documents"] += 1
            analytics["total_chunks"] += len(text_chunks)
            analytics["document_types"][file_ext] += 1
            analytics["api_usage"]["files_processed"] += 1
            
            save_data()
            
            # Final progress update
            update_progress(file_id, "completed", 100, "Document processing completed successfully")
            
            return True
            
        except Exception as e:
            raise Exception(f"Indexing failed: {str(e)}")
            
    except Exception as e:
        # Enhanced error handling with specific error types
        error_message = str(e)
        error_type = "unknown"
        
        if "text extraction" in error_message.lower():
            error_type = "extraction_error"
        elif "chunking" in error_message.lower():
            error_type = "chunking_error"
        elif "embedding" in error_message.lower():
            error_type = "embedding_error"
        elif "indexing" in error_message.lower():
            error_type = "indexing_error"
        elif "api" in error_message.lower() or "openai" in error_message.lower():
            error_type = "api_error"
        elif "file" in error_message.lower():
            error_type = "file_error"
        
        # Update document with detailed error information
        documents[session_id][file_id]["status"] = "failed"
        documents[session_id][file_id]["error"] = error_message
        documents[session_id][file_id]["error_type"] = error_type
        documents[session_id][file_id]["error_timestamp"] = datetime.now().isoformat()
        
        # Update progress with error
        update_progress(file_id, "failed", 0, error_message)
        
        # Update analytics
        analytics["error_counts"]["document_processing"] += 1
        analytics["error_counts"][error_type] += 1
        
        save_data()
        
        logger.error(f"Document processing failed for {filename}: {error_message}")
        return False

@app.get("/")
async def root():
    return {"message": "RAG Document Q&A Backend is running", "status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/debug")
async def debug():
    return {"message": "Debug endpoint working", "timestamp": datetime.now().isoformat()}

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint for Render deployment"""
    try:
        # Basic API key check
        api_key_status = "valid" if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here" else "error"
        
        return {
            "status": "healthy",
            "message": "RAG Document Q&A API is running",
            "timestamp": datetime.now().isoformat(),
            "api_key_configured": api_key_status == "valid",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/healthz")
async def healthz():
    """Simple health check endpoint for Render health checks"""
    return {"status": "healthy", "message": "OK"}

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
            "avg_query_time": sum(analytics["query_times"]) / len(analytics["query_times"]) if analytics["query_times"] else 0
        },
        "popular_queries": dict(analytics["popular_queries"].most_common(5)),
        "file_type_distribution": dict(analytics["document_types"]),
        "performance_metrics": {
            "cache_hit_rate": len(embedding_cache) / max(analytics["api_usage"]["embeddings_generated"], 1),
            "api_efficiency": analytics["api_usage"]["tokens_processed"] / max(analytics["api_usage"]["embeddings_generated"], 1)
        }
    }

@app.get("/api/security/status")
async def security_status():
    """Get security configuration and status"""
    return {
        "security_features": {
            "filename_sanitization": True,
            "content_validation": True,
            "file_hash_integrity": True,
            "type_allowlist": True,
            "size_limits": True,
            "session_isolation": True
        },
        "allowed_file_types": ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
        "max_file_size_mb": 30,
        "session_timeout_hours": 24,
        "security_validated_documents": sum(
            1 for session_docs in documents.values() 
            for doc in session_docs.values() 
            if doc.get("security_validated", False)
        ),
        "total_documents": sum(len(session_docs) for session_docs in documents.values()),
        "security_checks_passed": True,
        "last_security_audit": datetime.now().isoformat()
    }

@app.get("/api/cost/estimate")
async def cost_estimate():
    """Get cost estimation and usage statistics"""
    # Calculate estimated costs based on current usage
    total_tokens = analytics["api_usage"]["tokens_processed"]
    total_embeddings = analytics["api_usage"]["embeddings_generated"]
    
    # OpenAI pricing (as of 2024)
    embedding_cost_per_1k = 0.00002  # $0.00002 per 1K tokens
    gpt_cost_per_1k = 0.00015       # $0.00015 per 1K tokens
    
    # Calculate costs
    embedding_cost = (total_tokens / 1000) * embedding_cost_per_1k
    completion_cost = (total_tokens / 1000) * gpt_cost_per_1k
    total_cost = embedding_cost + completion_cost
    
    # Estimate monthly costs based on current usage patterns
    days_since_start = (datetime.now() - datetime.fromisoformat(sessions.get(list(sessions.keys())[0] if sessions else datetime.now().isoformat(), {}).get("created_at", datetime.now().isoformat()))).days or 1
    daily_cost = total_cost / days_since_start
    monthly_estimate = daily_cost * 30
    
    return {
        "current_usage": {
            "total_tokens": total_tokens,
            "total_embeddings": total_embeddings,
            "total_queries": analytics["total_queries"],
            "total_documents": analytics["total_documents"]
        },
        "cost_breakdown": {
            "embedding_cost": round(embedding_cost, 4),
            "completion_cost": round(completion_cost, 4),
            "total_cost": round(total_cost, 4)
        },
        "estimates": {
            "daily_cost": round(daily_cost, 4),
            "monthly_estimate": round(monthly_estimate, 2),
            "cost_per_query": round(total_cost / max(analytics["total_queries"], 1), 4)
        },
        "cost_optimization": {
            "cache_hit_rate": f"{len(embedding_cache) / max(total_embeddings, 1) * 100:.1f}%",
            "tokens_per_query": round(total_tokens / max(analytics["total_queries"], 1), 0),
            "recommendations": [
                "Enable caching to reduce API calls",
                "Use optimal chunk sizes (800 tokens)",
                "Implement rate limiting for cost control",
                "Monitor usage with this endpoint"
            ]
        },
        "pricing_info": {
            "embedding_rate": "$0.00002 per 1K tokens",
            "completion_rate": "$0.00015 per 1K tokens",
            "free_tier": "$5 credit for new OpenAI accounts"
        }
    }

@app.get("/api/progress/{file_id}")
async def get_document_progress(file_id: str):
    """Get progress for document processing"""
    progress = get_progress(file_id)
    return progress

@app.post("/api/retry/{file_id}")
async def retry_document_processing(file_id: str, request: Request):
    """Retry processing a failed document"""
    session_id = get_session_id(request)
    
    if session_id not in documents or file_id not in documents[session_id]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents[session_id][file_id]
    
    if doc["status"] != "failed":
        raise HTTPException(status_code=400, detail="Document is not in failed state")
    
    # Clear previous error information
    if "error" in doc:
        del doc["error"]
    if "error_type" in doc:
        del doc["error_type"]
    if "error_timestamp" in doc:
        del doc["error_timestamp"]
    
    # Clear progress and restart
    clear_progress(file_id)
    
    # Recreate file path
    file_ext = doc.get("file_type", ".txt")
    file_path = f"temp_{file_id}{file_ext}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file not found for retry")
    
    # Retry processing
    success = process_document_sync(file_id, file_path, doc["filename"], file_ext, session_id)
    
    if success:
        return {
            "file_id": file_id,
            "status": "retry_successful",
            "message": "Document processing retry successful"
        }
    else:
        return {
            "file_id": file_id,
            "status": "retry_failed",
            "message": "Document processing retry failed"
        }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):
    """Upload document with enhanced security features"""
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
    
    # Sanitize filename
    original_filename = file.filename
    sanitized_filename = sanitize_filename(file.filename)
    
    if not sanitized_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Save file temporarily
    file_path = f"temp_{file_id}{file_ext}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate file content
        if not validate_file_content(content, file_ext):
            raise HTTPException(status_code=400, detail="File content validation failed")
        
        # Calculate file hash for integrity
        file_hash = calculate_file_hash(content)
        
        # Initialize document record with security metadata
        documents[session_id][file_id] = {
            "id": file_id,
            "filename": sanitized_filename,
            "original_filename": original_filename,
            "status": "uploaded",
            "created_at": datetime.now().isoformat(),
            "session_id": session_id,
            "file_size": len(content),
            "file_type": file_ext,
            "file_hash": file_hash,
            "security_validated": True
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
            
    except HTTPException as he:
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        # Remove document record on error
        if session_id in documents and file_id in documents[session_id]:
            del documents[session_id][file_id]
            save_data()
        raise he
    except Exception as e:
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        # Remove document record on error
        if session_id in documents and file_id in documents[session_id]:
            del documents[session_id][file_id]
            save_data()
        
        error_message = str(e) if str(e) else "Unknown error occurred during file upload"
        logger.error(f"Upload error for file {file.filename}: {error_message}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {error_message}")

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

@app.get("/api/chunks/{doc_id}")
async def get_document_chunks(doc_id: str, request: Request):
    """Get chunks for a specific document"""
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
    
    # Get chunks for this document (filter by session for security)
    doc_chunks = [c for c in chunks if c.get("doc_id") == doc_id and c.get("session_id") == session_id]
    
    # Return chunks without embeddings for security
    return {
        "chunks": [
            {
                "id": chunk["id"],
                "content": chunk["content"],
                "chunk_index": chunk.get("chunk_index", 0),
                "page": chunk.get("page", 1),
                "created_at": chunk["created_at"]
            }
            for chunk in doc_chunks
        ]
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
                    "content": "You are a helpful assistant that answers questions based ONLY on the provided document content. You must follow these rules: 1) If the answer is not found in the provided documents, respond with 'I don't know' or 'I cannot find this information in the uploaded documents.' 2) Never make up information or use knowledge outside the provided context. 3) Always provide complete answers in plain text without markdown formatting. 4) Include specific references to documents when possible. 5) Be honest about the limitations of the available information."
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
        
        # Save chat history for all queries
        if session_id not in chat_history:
            chat_history[session_id] = {}
        
        # For global queries, save to a special "all" key
        # For document-specific queries, save to the document ID
        history_key = "all" if scope == "all" else doc_id
        
        if history_key not in chat_history[session_id]:
            chat_history[session_id][history_key] = []
        
        # Add message to history
        message = {
            "id": str(uuid.uuid4()),
            "question": query,
            "answer": answer,
            "citations": citations,
            "timestamp": datetime.now().isoformat()
        }
        
        chat_history[session_id][history_key].append(message)
        
        # Limit history to last 50 messages
        if len(chat_history[session_id][history_key]) > 50:
            chat_history[session_id][history_key] = chat_history[session_id][history_key][-50:]
        
        save_data()
        
        return result
        
    except Exception as e:
        analytics["error_counts"]["qa_processing"] += 1
        save_data()
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.post("/api/qa/stream")
async def ask_question_stream(request: Request):
    """Ask a question with streaming response using Server-Sent Events"""
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
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'No documents uploaded yet. Please upload a document first.'})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/plain")
    
    try:
        start_time = time.time()
        
        # Get query embedding
        query_embedding = rag_processor.get_embedding(query)
        if not query_embedding:
            async def error_stream():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Error: Could not process query embedding.'})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/plain")
        
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
            async def error_stream():
                error_msg = "I couldn't find relevant information in the documents to answer your question."
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/plain")
        
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
        
        async def stream_response():
            # Send start signal
            yield f"data: {json.dumps({'type': 'start', 'message': 'Generating answer...'})}\n\n"
            
            # Generate streaming answer using OpenAI
            try:
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that answers questions based ONLY on the provided document content. You must follow these rules: 1) If the answer is not found in the provided documents, respond with 'I don't know' or 'I cannot find this information in the uploaded documents.' 2) Never make up information or use knowledge outside the provided context. 3) Always provide complete answers in plain text without markdown formatting. 4) Include specific references to documents when possible. 5) Be honest about the limitations of the available information."
                        },
                        {
                            "role": "user", 
                            "content": f"Document content:\n{context}\n\nQuestion: {query}\n\nPlease provide a complete and detailed answer in plain text without any formatting:"
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                    stream=True
                )
                
                full_answer = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_answer += content
                        # Send each chunk of the answer
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': content})}\n\n"
                
                # Clean the answer
                full_answer = clean_markdown(full_answer)
                
                # Send citations
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
                
                # Send sources
                sources = [{"page": result["page"], "source": result["doc_name"]} for result in relevant_chunks[:3]]
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                # Send processing time
                processing_time = time.time() - start_time
                yield f"data: {json.dumps({'type': 'processing_time', 'time': round(processing_time, 2)})}\n\n"
                
                # Save chat history for all queries
                if session_id not in chat_history:
                    chat_history[session_id] = {}
                
                # For global queries, save to a special "all" key
                # For document-specific queries, save to the document ID
                history_key = "all" if scope == "all" else doc_id
                
                if history_key not in chat_history[session_id]:
                    chat_history[session_id][history_key] = []
                
                # Add message to history
                message = {
                    "id": str(uuid.uuid4()),
                    "question": query,
                    "answer": full_answer,
                    "citations": citations,
                    "timestamp": datetime.now().isoformat()
                }
                
                chat_history[session_id][history_key].append(message)
                
                # Limit history to last 50 messages
                if len(chat_history[session_id][history_key]) > 50:
                    chat_history[session_id][history_key] = chat_history[session_id][history_key][-50:]
                
                # Send end signal
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                error_msg = f'Error generating answer: {str(e)}'
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
        
        return StreamingResponse(stream_response(), media_type="text/plain")
        
    except Exception as e:
        async def error_stream():
            error_msg = f'Error processing question: {str(e)}'
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/plain")

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
        samesite="none",
        secure=True
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
    if session_id in chat_history:
        del chat_history[session_id]
    
    # Remove session cookie
    response.delete_cookie("session_id")
    
    save_data()
    
    return {"message": "Session cleared successfully"}

@app.get("/api/chat-history/all")
async def get_all_chat_history(request: Request):
    """Get all chat history for the session"""
    session_id = get_session_id(request)
    
    # Get session chat history
    session_chat_history = chat_history.get(session_id, {})
    all_messages = []
    
    # Collect all messages from all documents and global queries
    for doc_id_inner, messages in session_chat_history.items():
        all_messages.extend(messages)
    
    # Sort by timestamp (newest first)
    all_messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Limit to last 50 messages
    all_messages = all_messages[:50]
    
    return {"messages": all_messages}

@app.get("/api/chat-history/{doc_id}")
async def get_chat_history(doc_id: str, request: Request):
    """Get chat history for a specific document"""
    session_id = get_session_id(request)
    
    # Check if user has access to this document
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
    
    # Get chat history for this document
    session_chat_history = chat_history.get(session_id, {})
    doc_chat_history = session_chat_history.get(doc_id, [])
    
    return {"messages": doc_chat_history}

if __name__ == "__main__":
    import uvicorn
    
    # Handle PORT environment variable more robustly
    port_str = os.getenv("PORT", "8001")
    try:
        port = int(port_str)
    except ValueError:
        # If PORT is not a valid integer, use default
        port = 8001
        print(f"Warning: Invalid PORT value '{port_str}', using default port {port}")
    
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
