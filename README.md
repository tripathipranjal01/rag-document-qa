# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content using AI.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key

### First Time Setup

#### 1. Clone and Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd rag-document-qa

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 2. Install Backend Dependencies
```bash
cd backend

# Install Python dependencies
pip install --upgrade pip
pip install fastapi uvicorn openai python-multipart PyPDF2 python-docx tiktoken numpy Pillow pytesseract easyocr requests httpx

# Or install from requirements file
pip install -r requirements.txt
```

#### 3. Install Frontend Dependencies
```bash
cd ../frontend

# Install Node.js dependencies
npm install
```

#### 4. Configure OpenAI API Key
Edit `backend/main_simple.py` and replace the API key:
```python
api_key = "your_openai_api_key_here"
```

### Running the Application

#### Start Backend Server
```bash
cd backend
python main_simple.py
```
Backend will run on: http://localhost:8001

#### Start Frontend Server
```bash
cd frontend
npm run dev
```
Frontend will run on: http://localhost:3000

## 📁 Project Structure

```
rag-document-qa/
├── backend/
│   ├── main_simple.py      # Main backend server
│   ├── requirements.txt    # Python dependencies
│   └── uploads/           # Document storage
├── frontend/
│   ├── app/              # Next.js application
│   └── package.json      # Node.js dependencies
├── README.md             # This file
└── ARCHITECTURE.md       # System architecture
```

## 🔧 Available Backend Versions

- `main_simple.py` - Basic RAG implementation (recommended)
- `main_minimal.py` - Minimal version
- `main_working.py` - Working version with enhanced features
- `main_enhanced.py` - Enhanced version
- `main_advanced.py` - Advanced version
- `main_final.py` - Final version

## 🌐 API Endpoints

- `GET /` - Health check
- `GET /api/health` - Detailed health status
- `POST /api/upload` - Upload documents
- `GET /api/documents` - List documents
- `POST /api/qa` - Ask questions
- `GET /api/analytics` - Get usage analytics

## 📄 Supported File Types

- PDF (.pdf)
- Word documents (.docx)
- Text files (.txt)
- Images (.png, .jpg, .jpeg, .bmp, .tiff)

## 🔒 Environment Variables

Set these environment variables or update them in the code:

```bash
OPENAI_API_KEY=your_openai_api_key_here
PORT=8001  # Backend port
```

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies
   ```bash
   pip install <missing_module>
   ```

2. **Tiktoken build error**: Update pip first
   ```bash
   pip install --upgrade pip
   pip install tiktoken --no-build-isolation
   ```

3. **Port already in use**: Change port in main_simple.py
   ```python
   port = int(os.getenv("PORT", 8002))  # Change to different port
   ```

## 📊 Features

- Document upload and processing
- Text extraction from multiple file types
- Semantic search using embeddings
- AI-powered Q&A with citations
- Session management
- Analytics and usage tracking
- Document sharing and collaboration
- OCR for image files

## 🏗️ Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system architecture and design decisions.
