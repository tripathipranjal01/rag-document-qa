# RAG Document Q&A System

A production-ready RAG (Retrieval-Augmented Generation) application for document analysis and Q&A. Built with FastAPI backend and Next.js frontend, featuring real-time streaming responses, document processing, and intelligent citation management.

## Live Demo

- **Frontend**: https://rag-frontend-0ne1.onrender.com
- **Backend API**: https://rag-backend-4ca4.onrender.com
- **Health Check**: https://rag-backend-4ca4.onrender.com/api/health

## Core Features

- **Document Processing**: PDF, DOCX, TXT support with incremental ingestion
- **Intelligent Q&A**: Two scopes - "This document" vs "All documents"
- **Real-time Streaming**: Server-Sent Events for live response streaming
- **Smart Citations**: Clickable citations with page references and snippets
- **Session Management**: Isolated user sessions with persistent storage
- **Progress Tracking**: Real-time document processing status
- **Security**: File validation, sanitization, and secure API handling

## Architecture Overview

### Stack Choice & Reasoning

**Backend: FastAPI + Python 3.13**
- Fast async performance for streaming responses
- Excellent OpenAI integration
- Built-in security features and validation
- Simple deployment and scaling

**Frontend: Next.js 14 + TypeScript**
- Server-side rendering for better SEO
- Type safety for maintainable code
- Modern React features with hooks
- Optimized for production deployment

**AI Provider: OpenAI**
- GPT-4o-mini for cost-effective completions
- text-embedding-3-small for semantic search
- Reliable API with good rate limits
- Strong hallucination controls

**Deployment: Render.com**
- Free tier available for development
- Simple GitHub integration
- Automatic scaling and SSL
- Good performance for small to medium workloads

### Ingestion & Incrementality

**Trigger Mechanism:**
- File upload triggers async processing pipeline
- No manual reindexing required
- New documents become searchable within 60 seconds

**Processing Pipeline:**
1. **Upload** - File validation and storage
2. **Extract** - Text extraction (PDF/DOCX/TXT/Images)
3. **Chunk** - 800-token chunks with 150-token overlap
4. **Embed** - OpenAI text-embedding-3-small
5. **Index** - In-memory storage with pickle persistence

**Real-time Updates:**
- WebSocket-style progress tracking
- Live status updates in UI
- Automatic document list refresh

### Chunking Strategy

**Configuration:**
- **Chunk Size**: 800 tokens (optimal for GPT-4o-mini context)
- **Overlap**: 150 tokens (prevents information loss at boundaries)
- **Method**: TikToken-based tokenization for accuracy
- **Top-k Retrieval**: 8 results (balanced recall vs precision)

**Rationale:**
- 800 tokens provide sufficient context while staying within limits
- 150-token overlap ensures no important information is lost
- TikToken gives exact token counts matching OpenAI models

### Prompting Strategy

**System Prompt Design:**
```
You are a helpful assistant that answers questions based ONLY on the provided document content. 
You must follow these rules:
1) If the answer is not found in the provided documents, respond with 'I don't know' or 'I cannot find this information in the uploaded documents.'
2) Never make up information or use knowledge outside the provided context.
3) Always provide complete answers in plain text without markdown formatting.
4) Include specific references to documents when possible.
5) Be honest about the limitations of the available information.
```

**Context Formatting:**
- Documents clearly labeled with names
- Chunk content preserved with source attribution
- Page numbers maintained for accurate citations

**Hallucination Controls:**
- Explicit instructions to stay within provided context
- "I don't know" responses for unanswerable questions
- Temperature set to 0.3 for consistent, factual responses
- Max tokens limited to prevent rambling

### Failure Modes Handled

**Bad PDFs:**
- Magic number validation
- Content extraction fallbacks
- Error reporting with retry options

**Empty Files:**
- File size validation (minimum requirements)
- Content validation before processing
- Clear error messages to users

**Timeouts:**
- Async processing with progress tracking
- Request timeout handling
- Background job management

**Rate Limits:**
- OpenAI API rate limit handling
- Exponential backoff for retries
- Request queuing for high load

**Network Issues:**
- Graceful DNS error handling
- Connection retry logic
- Fallback error messages

### System Limits

**File Restrictions:**
- **Types**: PDF, DOCX, TXT (images disabled for free tier)
- **Size**: 30MB maximum (configurable)
- **Pages**: No explicit limit (tested with 100+ page PDFs)

**Rate Limits:**
- OpenAI API: Standard tier limits
- File uploads: 10 concurrent per session
- Query frequency: No artificial limits

**Reasoning:**
- 30MB covers most business documents
- PDF/DOCX/TXT handle 90% of use cases
- Image OCR requires paid hosting for stability

## Local Development Setup

### Prerequisites

- Python 3.13+
- Node.js 18+
- OpenAI API key
- Git

### Backend Setup

1. **Clone and setup backend:**
```bash
git clone https://github.com/tripathipranjal01/rag-document-qa.git
cd rag-document-qa/backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "PORT=8001" >> .env
echo "HOST=0.0.0.0" >> .env
echo "ALLOWED_ORIGINS=http://localhost:3000" >> .env
```

5. **Run backend:**
```bash
python main_simple.py
```

Backend available at: http://localhost:8001

### Frontend Setup

1. **Setup frontend:**
```bash
cd ../frontend
npm install
```

2. **Configure environment:**
```bash
# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8001" > .env.local
```

3. **Run frontend:**
```bash
npm run dev
```

Frontend available at: http://localhost:3000

### Testing the Setup

1. Open http://localhost:3000
2. Upload a PDF/DOCX/TXT file
3. Wait for processing to complete
4. Ask a question about the document
5. Verify citations and responses work

## OCR Support (Local Development Only)

The system includes OCR functionality for image processing, but it's disabled in production due to memory constraints on free hosting tiers.

### Enabling OCR Locally

1. **Update requirements.txt:**
```bash
# Add to backend/requirements.txt
numpy>=1.26.0
easyocr==1.7.0
```

2. **Install OCR dependencies:**
```bash
# For Linux/Mac
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract  # macOS

# For Windows
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
```

3. **Enable OCR in code:**
```python
# In backend/main_simple.py, line 21:
# Change from:
# import easyocr  # Temporarily disabled for faster deployment

# To:
import easyocr
```

4. **Update OCR function:**
```python
# In extract_text_from_image function, replace the simplified version with:
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
```

5. **Restart backend:**
```bash
python main_simple.py
```

### OCR File Support

Once enabled, the system supports:
- PNG, JPG, JPEG, BMP, TIFF files
- Text extraction from images
- Dual OCR engine fallback (EasyOCR -> Tesseract)

### Why OCR is Disabled in Production

- **Memory Requirements**: EasyOCR needs 1GB+ RAM (free tier: 512MB)
- **Build Size**: OCR dependencies add 1.3GB+ to deployment
- **Cold Start**: 30-60 second model loading on first request
- **Cost**: Requires paid hosting plans for stable operation

## Production Deployment

### Backend Deployment (Render)

1. **Create Render account** at render.com
2. **Connect GitHub repository**
3. **Create new Web Service**
4. **Configure service:**
   - **Name**: rag-backend
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main_simple.py`
   - **Environment**: Python 3

5. **Set environment variables:**
```
OPENAI_API_KEY=your_actual_openai_api_key
PORT=10000
HOST=0.0.0.0
ALLOWED_ORIGINS=https://your-frontend-domain.onrender.com
```

6. **Deploy service**

### Frontend Deployment (Render)

1. **Create new Static Site**
2. **Configure:**
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/out`

3. **Set environment variables:**
```
NEXT_PUBLIC_API_URL=https://your-backend-domain.onrender.com
```

4. **Deploy service**

## Cost Estimation

### OpenAI API Costs (Monthly)

**Light Usage (10 documents, 50 queries):**
- Embeddings: ~$0.50
- Completions: ~$2.00
- **Total**: ~$2.50/month

**Medium Usage (100 documents, 500 queries):**
- Embeddings: ~$5.00
- Completions: ~$20.00
- **Total**: ~$25.00/month

**Heavy Usage (1000 documents, 5000 queries):**
- Embeddings: ~$50.00
- Completions: ~$200.00
- **Total**: ~$250.00/month

### Hosting Costs

**Free Tier (Render):**
- Backend: Free (with limitations)
- Frontend: Free
- **Total**: $0/month

**Paid Tier (Render Starter):**
- Backend: $7/month
- Frontend: Free
- **Total**: $7/month

### Scaling Considerations

- Costs scale linearly with document count and query volume
- Embedding costs are one-time per document
- Query costs depend on response length and frequency
- Free tier suitable for development and small projects

## API Documentation

### Authentication
- Session-based with secure cookies
- No registration required
- Automatic session creation

### Core Endpoints

**Health & Status:**
- `GET /` - Basic health check
- `GET /api/health` - Detailed system status
- `GET /api/session` - Get current session info
- `DELETE /api/session` - Clear session

**Document Management:**
- `POST /api/upload` - Upload document (multipart/form-data)
- `GET /api/documents` - List user documents
- `GET /api/documents/{doc_id}/content` - Get document content
- `DELETE /api/documents/{doc_id}` - Delete document
- `GET /api/progress/{file_id}` - Check processing status

**Q&A System:**
- `POST /api/qa` - Synchronous question answering
- `POST /api/qa/stream` - Streaming Q&A with SSE
- `GET /api/chunks/{doc_id}` - Get document chunks

### Request Examples

**Upload Document:**
```bash
curl -X POST "http://localhost:8001/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Ask Question:**
```bash
curl -X POST "http://localhost:8001/api/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "scope": "all",
    "doc_id": null
  }'
```

## Project Structure

```
rag-document-qa/
├── backend/
│   ├── main_simple.py          # Main FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── document_processor.py   # Document processing utilities
│   ├── rag_service.py         # RAG-specific services
│   └── models.py              # Data models
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # Main React component
│   │   ├── globals.css        # Global styles
│   │   └── layout.tsx         # App layout
│   ├── package.json           # Node.js dependencies
│   └── next.config.ts         # Next.js configuration
├── tests/
│   └── test_qa_integration.py # Integration tests
├── ARCHITECTURE.md            # System architecture details
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # This file
```

## Known Limitations

### Current Limitations

1. **Image OCR**: Disabled on free hosting tier due to memory constraints
2. **File Storage**: In-memory with pickle persistence (not suitable for high scale)
3. **Concurrent Users**: Limited by single-instance deployment
4. **Search Features**: No full-text search across documents

### Future Enhancements

1. **Database Integration**: PostgreSQL with pgvector for scalability
2. **Advanced OCR**: Cloud-based OCR services for production
3. **Multi-tenant Support**: User authentication and data isolation
4. **Advanced Analytics**: Usage metrics and performance monitoring
5. **Document Versioning**: Support for document updates and history

## Troubleshooting

### Common Issues

**Backend won't start:**
- Check OPENAI_API_KEY in environment variables
- Verify Python 3.13+ is installed
- Check port 8001 is available

**Frontend connection errors:**
- Verify NEXT_PUBLIC_API_URL points to backend
- Check CORS settings in backend
- Ensure backend is running and accessible

**File upload failures:**
- Check file size (30MB limit)
- Verify file type is supported
- Check backend logs for detailed errors

**OCR not working:**
- Ensure tesseract is installed locally
- Check EasyOCR dependencies are installed
- Verify sufficient system memory (1GB+)

### Debug Commands

```bash
# Check backend health
curl http://localhost:8001/api/health

# View backend logs
tail -f backend.log

# Test file upload
curl -X POST http://localhost:8001/api/upload -F "file=@test.pdf"

# Check frontend build
npm run build
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push to branch: `git push origin feature-name`
6. Create a Pull Request

### Development Guidelines

- Follow TypeScript/Python type hints
- Add tests for new features
- Update documentation
- Ensure compatibility with deployment platforms
- Test with various document types and sizes

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions:

1. Check this README and troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description
4. Include system info, error logs, and reproduction steps

---

**Built with FastAPI, Next.js, and OpenAI • Deployed on Render • Ready for production**