from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if database is configured
DATABASE_URL = os.getenv("DATABASE_URL", "")
DEMO_MODE = not DATABASE_URL or DATABASE_URL == "your_supabase_connection_string_here"

# Import database modules
try:
    from sqlalchemy.orm import Session
    from database import get_db, create_tables
    from models import Document, DocumentChunk, Base
except ImportError:
    Session = None
    get_db = None
    create_tables = None
    Document = None
    DocumentChunk = None
    Base = None

from document_processor import DocumentProcessor
from rag_service import RAGService

app = FastAPI(title="RAG Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
processor = DocumentProcessor()
rag_service = RAGService()

# Demo storage (in-memory for testing)
demo_documents = []
demo_chunks = []

@app.on_event("startup")
async def startup_event():
    os.makedirs("uploads", exist_ok=True)
    # Initialize DB tables if DB configured
    if not DEMO_MODE and create_tables:
        try:
            create_tables()
        except Exception as e:
            print(f"Startup DB init skipped: {e}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "RAG API is running", "demo_mode": DEMO_MODE}

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # Get database session if available
    db = None
    if not DEMO_MODE and get_db:
        try:
            db = next(get_db())
        except:
            pass
    
    # Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_types = ['.pdf', '.docx', '.txt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    max_size = int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024
    if file.size > max_size:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Save file
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}"
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    if DEMO_MODE:
        # Demo mode - store in memory
        demo_doc = {
            "id": file_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": len(content),
            "file_type": file_ext,
            "status": "uploaded",
            "created_at": "2024-01-01T00:00:00Z"
        }
        demo_documents.append(demo_doc)
        
        # Process document in background
        background_tasks.add_task(process_document_background_demo, file_id, file_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "File uploaded successfully, processing in background (DEMO MODE)"
        }
    else:
        # Production mode with database
        if db and Document:
            db_document = Document(
                id=file_id,
                filename=file.filename,
                file_path=file_path,
                file_size=len(content),
                file_type=file_ext,
                status="uploaded"
            )
            db.add(db_document)
            db.commit()
            
            # Process document in background
            background_tasks.add_task(process_document_background, file_id, file_path, db)
        else:
            # Fallback to demo mode
            demo_doc = {
                "id": file_id,
                "filename": file.filename,
                "file_path": file_path,
                "file_size": len(content),
                "file_type": file_ext,
                "status": "uploaded",
                "created_at": "2024-01-01T00:00:00Z"
            }
            demo_documents.append(demo_doc)
            background_tasks.add_task(process_document_background_demo, file_id, file_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "File uploaded successfully, processing in background"
        }

async def process_document_background_demo(file_id: str, file_path: str):
    """Demo background processing without database"""
    try:
        # Update status to processing
        for doc in demo_documents:
            if doc["id"] == file_id:
                doc["status"] = "processing"
                break
        
        # Process document
        doc_data = processor.process_document(file_path)
        
        # Chunk document
        chunks = processor.chunk_text(doc_data['content'])
        
        # Store chunks with embeddings (demo mode)
        for i, chunk in enumerate(chunks):
            try:
                embedding = rag_service.create_embedding(chunk['text'])
                
                demo_chunk = {
                    "id": str(uuid.uuid4()),
                    "document_id": file_id,
                    "chunk_text": chunk['text'],
                    "chunk_index": i,
                    "page_number": chunk['page'],
                    "embedding": embedding,
                    "metadata": {
                        'word_count': chunk['word_count'],
                        'start_word': chunk['start_word'],
                        'end_word': chunk['end_word']
                    }
                }
                demo_chunks.append(demo_chunk)
            except Exception as e:
                print(f"Error creating embedding for chunk {i}: {e}")
        
        # Update status to indexed
        for doc in demo_documents:
            if doc["id"] == file_id:
                doc["status"] = "indexed"
                break
        
    except Exception as e:
        for doc in demo_documents:
            if doc["id"] == file_id:
                doc["status"] = "failed"
                break
        print(f"Error processing document {file_id}: {str(e)}")

async def process_document_background(file_id: str, file_path: str, db: Session):
    """Production background processing with database"""
    try:
        # Update status to processing
        document = db.query(Document).filter(Document.id == file_id).first()
        document.status = "processing"
        db.commit()
        
        # Process document
        doc_data = processor.process_document(file_path)
        
        # Chunk document
        chunks = processor.chunk_text(doc_data['content'])
        
        # Store chunks with embeddings
        for i, chunk in enumerate(chunks):
            embedding = rag_service.create_embedding(chunk['text'])
            
            db_chunk = DocumentChunk(
                document_id=file_id,
                chunk_text=chunk['text'],
                chunk_index=i,
                page_number=chunk['page'],
                embedding=embedding,  # Store as vector
                chunk_metadata={
                    'word_count': chunk['word_count'],
                    'start_word': chunk['start_word'],
                    'end_word': chunk['end_word']
                }
            )
            db.add(db_chunk)
        
        db.commit()
        
        # Update status to indexed
        document.status = "indexed"
        db.commit()
        
    except Exception as e:
        document = db.query(Document).filter(Document.id == file_id).first()
        document.status = "failed"
        db.commit()
        print(f"Error processing document {file_id}: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    if DEMO_MODE:
        return demo_documents
    else:
        # Try to get database session
        db = None
        if get_db:
            try:
                db = next(get_db())
            except:
                pass
        
        if db and Document:
            documents = db.query(Document).order_by(Document.created_at.desc()).all()
            return [
                {
                    "id": str(doc.id),
                    "filename": doc.filename,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat()
                }
                for doc in documents
            ]
        else:
            return demo_documents

@app.post("/api/qa")
async def ask_question(request: dict):
    query = request.get("question")
    document_id = request.get("document_id")
    scope = request.get("scope", "all")
    
    if not query:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        if DEMO_MODE:
            # Demo mode - use in-memory chunks
            chunks = demo_chunks
            if scope == "document" and document_id:
                chunks = [c for c in demo_chunks if c["document_id"] == document_id]
            
            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "citations": [],
                    "sources": []
                }
            
            # Convert to dict format for RAG service
            chunk_dicts = [
                {
                    'text': chunk['chunk_text'],
                    'page': chunk['page_number'],
                    'metadata': chunk['metadata']
                }
                for chunk in chunks[:8]  # Limit to 8 chunks
            ]
        else:
            # Try to get database session
            db = None
            if get_db:
                try:
                    db = next(get_db())
                except:
                    pass
            
            if db and rag_service:
                # Production mode with vector search
                query_embedding = rag_service.create_embedding(query)
                
                # Search for similar chunks
                chunks = rag_service.search_similar_chunks(
                    query_embedding=query_embedding,
                    document_id=document_id if scope == "document" else None,
                    top_k=8 if scope == "document" else 12,
                    db=db
                )
                
                if not chunks:
                    return {
                        "answer": "I couldn't find any relevant information in the documents to answer your question.",
                        "citations": [],
                        "sources": []
                    }
                
                # Convert to dict format for RAG service
                chunk_dicts = [
                    {
                        'text': chunk['text'],
                        'page': chunk['page'],
                        'metadata': chunk['metadata']
                    }
                    for chunk in chunks
                ]
            else:
                # Fallback to demo mode
                chunks = demo_chunks
                if scope == "document" and document_id:
                    chunks = [c for c in demo_chunks if c["document_id"] == document_id]
                
                if not chunks:
                    return {
                        "answer": "I couldn't find any relevant information in the documents to answer your question.",
                        "citations": [],
                        "sources": []
                    }
                
                chunk_dicts = [
                    {
                        'text': chunk['chunk_text'],
                        'page': chunk['page_number'],
                        'metadata': chunk['metadata']
                    }
                    for chunk in chunks[:8]
                ]
        
        # Generate answer
        answer = rag_service.generate_answer(query, chunk_dicts)
        
        # Prepare citations
        citations = []
        for chunk in (chunks[:3] if DEMO_MODE else chunks[:3]):
            text = chunk['chunk_text'] if DEMO_MODE else chunk['text']
            page = chunk['page_number'] if DEMO_MODE else chunk['page']
            citations.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "page": page,
            })
        
        return {
            "answer": answer,
            "citations": citations,
            "sources": [{"page": chunk['page_number'] if DEMO_MODE else chunk['page']} for chunk in chunks[:3]]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
