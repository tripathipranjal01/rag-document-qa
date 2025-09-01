# Document Q&A

A RAG application for document analysis and Q&A, built with FastAPI backend and Next.js frontend.

## Features

- Document Processing: Support for PDF, DOCX, TXT, and images (with OCR)
- Q&A: Ask questions about your documents with grounded answers
- Streaming Responses: Real-time streaming answers using Server-Sent Events
- Chat History: Per document or across all documents
- Security: File validation, sanitization, and secure API key management
- Progress Tracking: Real-time document processing status
- Production Ready: Optimized for deployment on Render, Vercel, or any cloud platform

## Architecture

- Backend: FastAPI with Python 3.13
- Frontend: Next.js 14 with TypeScript
- AI: OpenAI GPT-4o-mini for embeddings and completions
- Storage: In-memory with pickle persistence (session-based)
- Deployment: Render.com (backend) + Vercel/Render (frontend)

## Prerequisites

- Python 3.13+
- Node.js 18+
- OpenAI API key
- Render.com account (for deployment)

## Local Development Setup

### Backend Setup

1. **Clone and navigate to backend:**
```bash
cd backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp env.example .env
# Edit .env with your OpenAI API key
```

5. **Run the backend:**
```bash
python main_simple.py
```

Backend will be available at `http://localhost:8001`

### Frontend Setup

1. **Navigate to frontend:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Set up environment variables:**
```bash
cp env.example .env.local
# Edit .env.local with your backend URL
```

4. **Run the frontend:**
```bash
npm run dev
```

Frontend will be available at `http://localhost:3000`

## Production Deployment

### Backend Deployment (Render)

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure the service:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main_simple.py`
   - **Environment**: Python 3

4. **Set Environment Variables in Render Dashboard:**
   ```
   OPENAI_API_KEY=your_actual_openai_api_key
   PORT=8001
   HOST=0.0.0.0
   ALLOWED_ORIGINS=https://your-frontend-domain.onrender.com
   MAX_FILE_SIZE=52428800
   ENABLE_FILE_VALIDATION=true
   ENABLE_FILENAME_SANITIZATION=true
   LOG_LEVEL=INFO
   ```

5. **Deploy the service**

### Frontend Deployment (Render/Vercel)

1. **Connect your GitHub repository to Render/Vercel**
2. **Create a new Static Site (Render) or Next.js App (Vercel)**
3. **Set Environment Variables:**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-domain.onrender.com
   ```

4. **Deploy the service**

## Environment Variables

### Backend (.env)
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
PORT=8001
HOST=0.0.0.0
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend-domain.onrender.com
MAX_FILE_SIZE=52428800
ENABLE_FILE_VALIDATION=true
ENABLE_FILENAME_SANITIZATION=true
LOG_LEVEL=INFO
```

### Frontend (.env.local)
```bash
# Required for production
NEXT_PUBLIC_API_URL=https://your-backend-domain.onrender.com

# Optional for development
# NEXT_PUBLIC_API_URL=http://localhost:8001
```

## API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /api/health` - Detailed health status
- `GET /api/session` - Get/create session
- `DELETE /api/session` - Clear session

### Document Management
- `GET /api/documents` - List documents
- `POST /api/upload` - Upload document
- `GET /api/documents/{doc_id}/content` - Get document content
- `GET /api/chunks/{doc_id}` - Get document chunks
- `DELETE /api/documents/{doc_id}` - Delete document

### Q&A Endpoints
- `POST /api/qa` - Synchronous Q&A
- `POST /api/qa/stream` - Streaming Q&A (SSE)
- `GET /api/chat-history/{doc_id}` - Document-specific chat history
- `GET /api/chat-history/all` - Global chat history

### Monitoring & Security
- `GET /api/security/status` - Security status
- `GET /api/cost/estimate` - Cost estimation
- `GET /api/progress/{file_id}` - Processing progress
- `POST /api/retry/{file_id}` - Retry failed processing

## Security Features

- **File Validation**: Magic number checking and content validation
- **Filename Sanitization**: Prevents path traversal attacks
- **CORS Configuration**: Proper origin validation
- **API Key Management**: Secure environment variable handling
- **File Size Limits**: Configurable upload limits
- **Content Type Validation**: Whitelist of allowed file types

## Performance & Monitoring

- **Response Time Tracking**: P50, P95, P99 metrics
- **Cost Estimation**: OpenAI API usage tracking
- **Progress Tracking**: Real-time document processing status
- **Error Handling**: Comprehensive error reporting and retry mechanisms

## Usage

1. **Upload Documents**: Drag and drop or click to upload PDF, DOCX, TXT, or image files
2. **Select Document**: Choose a specific document or "All Documents" for global search
3. **Ask Questions**: Type your question and get AI-powered answers with citations
4. **View Sources**: Click on citations to jump to the relevant document sections
5. **Chat History**: View conversation history per document or globally

## Development

### Project Structure
```
├── backend/
│   ├── main_simple.py          # Main FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── env.example            # Environment variables template
│   └── render.yaml            # Render deployment config
├── frontend/
│   ├── app/
│   │   └── page.tsx           # Main React component
│   ├── package.json           # Node.js dependencies
│   ├── env.example           # Environment variables template
│   └── render.yaml           # Render deployment config
└── README.md                 # This file
```

### Adding New Features
1. Backend: Add endpoints in `main_simple.py`
2. Frontend: Update components in `app/page.tsx`
3. Environment: Update `.env.example` files
4. Documentation: Update this README

## Troubleshooting

### Common Issues

1. **Backend not starting**: Check OpenAI API key in environment variables
2. **CORS errors**: Verify `ALLOWED_ORIGINS` includes your frontend URL
3. **File upload fails**: Check file size limits and supported formats
4. **Streaming not working**: Ensure SSE endpoint is accessible

### Debug Mode
- Backend: Check logs in Render dashboard
- Frontend: Open browser developer tools
- API: Use `/api/health` endpoint for status

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub
