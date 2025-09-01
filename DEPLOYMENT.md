# RAG Document Q&A - Deployment Guide

## Local Development

### Backend
```bash
# Set environment variables
export OPENAI_API_KEY="your_key_here"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python backend/main_simple.py
```

### Frontend
```bash
# Install dependencies
cd frontend
npm install

# Run frontend
npm run dev
```

## Render Deployment

### 1. Fork this repository to your GitHub account

### 2. Create Render services:

**Backend:**
- Connect your GitHub repo
- Select "rag-document-qa-main" repository
- Environment: Python
- Build Command: `pip install -r requirements.txt`
- Start Command: `python backend/main_simple.py`

**Frontend:**
- Connect your GitHub repo  
- Environment: Node
- Build Command: `cd frontend && npm install && npm run build`
- Start Command: `cd frontend && npm start`

### 3. Set Environment Variables in Render Dashboard:

**Backend:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `ALLOWED_ORIGINS`: https://your-frontend-name.onrender.com,http://localhost:3000
- `PORT`: 10000 (Render sets this automatically)
- `HOST`: 0.0.0.0

**Frontend:**
- `NEXT_PUBLIC_API_URL`: https://your-backend-name.onrender.com

### 4. Update service names in render.yaml:
- Replace `your-rag-backend` with your actual backend service name
- Replace `your-rag-frontend` with your actual frontend service name
- Replace `your-frontend-name.onrender.com` with your actual frontend URL

### 5. Deploy:
- Push to GitHub
- Render will auto-deploy both services

## Environment Variables Summary

### Required:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ALLOWED_ORIGINS`: Frontend domain for CORS
- `NEXT_PUBLIC_API_URL`: Backend URL for frontend

### Optional:
- `MAX_FILE_SIZE`: File upload limit (default: 50MB)
- `LOG_LEVEL`: Logging level (default: INFO)

## Features

- ✅ PDF, DOCX, TXT file upload
- ✅ AI-powered document analysis  
- ✅ Question answering with citations
- ✅ Session management
- ✅ Real-time processing
- ✅ Cross-origin support
- ❌ OCR (disabled for deployment compatibility)

## Troubleshooting

### Documents not appearing:
1. Check CORS configuration
2. Verify environment variables
3. Clear browser cache
4. Check browser DevTools for errors

### Upload failures:
1. Verify OpenAI API key
2. Check file size limits
3. Ensure supported file types
