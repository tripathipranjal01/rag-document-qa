from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import openai
import PyPDF2
import re
from docx import Document as DocxDocument

# Set OpenAI API key
openai.api_key = "your_openai_api_key_here"

app = FastAPI(title="RAG Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
documents = []
document_contents = {}

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

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "RAG API is running"}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
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
        
        # Extract text from file
        text_content = extract_text_from_file(file_path, file_ext)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content found in the file")
        
        # Create document record
        doc = {
            "id": file_id,
            "filename": file.filename,
            "status": "indexed",
            "created_at": "2024-01-01T00:00:00Z"
        }
        documents.append(doc)
        document_contents[file_id] = text_content
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "indexed",
            "message": "File uploaded and processed successfully"
        }
    
    except Exception as e:
        # Clean up file if it was created
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    return documents

@app.post("/api/qa")
async def ask_question(request: dict):
    query = request.get("question", "")
    if not query:
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not document_contents:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "citations": [],
            "sources": []
        }
    
    try:
        # Simple approach: use all document content as context
        all_content = "\n\n".join(document_contents.values())
        
        # Generate answer using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document content. Always provide complete and detailed answers without using markdown formatting. Do not use asterisks, bold, or any special formatting. Provide clean, plain text answers."},
                {"role": "user", "content": f"Document content:\n{all_content}\n\nQuestion: {query}\n\nPlease provide a complete and detailed answer in plain text without any formatting:"}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # Clean the answer to remove any markdown formatting
        answer = clean_markdown(answer)
        
        # Simple citations
        citations = []
        for doc_id, content in document_contents.items():
            doc_name = next((d["filename"] for d in documents if d["id"] == doc_id), "Unknown")
            citations.append({
                "text": content[:200] + "..." if len(content) > 200 else content,
                "page": 1,
                "source": doc_name
            })
        
        return {
            "answer": answer,
            "citations": citations[:3],
            "sources": [{"page": 1, "source": doc["filename"]} for doc in documents[:3]]
        }
        
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "citations": [],
            "sources": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
