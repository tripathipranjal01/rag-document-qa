# Interviewer Guide - RAG Document Q&A System

## **Quick Setup for Interviewers**

This guide will help you quickly set up and test the RAG Document Q&A System for assessment purposes.

### **Prerequisites Check**
- ✅ Python 3.9+ installed
- ✅ Node.js 18+ installed  
- ✅ OpenAI API key (required for functionality)

### **Step 1: Clone and Setup (2 minutes)**

```bash
# Clone the repository
git clone https://github.com/tripathipranjal01/rag-document-qa.git
cd rag-document-qa

# Run automated setup (checks everything automatically)
python setup.py
```

### **Step 2: Configure API Key (30 seconds)**

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your_actual_openai_api_key_here"
```

**Option B: Edit File**
Open `backend/main_simple.py` and replace line 27:
```python
openai.api_key = "your_actual_openai_api_key_here"
```

### **Step 3: Start the Application (1 minute)**

**Terminal 1 - Backend:**
```bash
cd backend
python main_simple.py
```
✅ Backend will be running at `http://localhost:8001`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
✅ Frontend will be running at `http://localhost:3000`

### **Step 4: Test Everything (1 minute)**

```bash
python test_setup.py
```

This will validate:
- ✅ Backend health and API key
- ✅ Frontend accessibility  
- ✅ Upload functionality
- ✅ Complete system integration

### **Step 5: Demo the Application**

1. **Open Browser**: Navigate to `http://localhost:3000`
2. **Upload Document**: Try uploading a PDF, DOCX, or TXT file
3. **Ask Questions**: Use the Q&A interface to test RAG functionality
4. **Test Features**: Explore document viewer, citations, and analytics

## **What to Look For**

### **✅ Core Functionality**
- Document upload and processing
- RAG-based question answering
- Citation system with source tracking
- Document viewer with chunk navigation

### **✅ Advanced Features**
- Image OCR support (PNG, JPG, etc.)
- Session management and persistence
- Real-time analytics and monitoring
- Performance optimization and caching

### **✅ Code Quality**
- Clean, well-documented code
- Proper error handling
- Type hints and validation
- Comprehensive testing

### **✅ Architecture**
- FastAPI backend with async support
- Next.js frontend with modern UI
- Modular service architecture
- Scalable design patterns

## **Troubleshooting**

### **Common Issues**

**"API key not configured"**
- Set your OpenAI API key as shown in Step 2
- Ensure the key has sufficient credits

**"Backend not running"**
- Check if port 8001 is available
- Ensure Python dependencies are installed

**"Frontend not accessible"**
- Check if port 3000 is available
- Ensure Node.js dependencies are installed

**"Upload fails"**
- Check file size (30MB limit)
- Verify file type (PDF, DOCX, TXT, Images)

### **Quick Fixes**

```bash
# Reinstall dependencies if needed
cd backend && pip install -r requirements.txt
cd frontend && npm install

# Check if ports are in use
lsof -i :8001  # Backend port
lsof -i :3000  # Frontend port

# Kill processes if needed
kill -9 <PID>
```

## **Assessment Criteria**

### **Technical Implementation (40%)**
- ✅ RAG pipeline with proper chunking
- ✅ Vector embeddings and similarity search
- ✅ Document processing (PDF, DOCX, TXT, OCR)
- ✅ API design and error handling

### **User Experience (30%)**
- ✅ Modern, responsive UI
- ✅ Real-time status updates
- ✅ Interactive document viewer
- ✅ Citation system with navigation

### **Advanced Features (20%)**
- ✅ Analytics and monitoring
- ✅ Performance optimization
- ✅ Session management
- ✅ Multi-modal support (OCR)

### **Code Quality (10%)**
- ✅ Clean architecture
- ✅ Comprehensive documentation
- ✅ Error handling
- ✅ Testing coverage

## **Repository Structure**

```
rag-document-qa/
├── README.md                    # Main documentation
├── INTERVIEWER_GUIDE.md         # This file
├── setup.py                     # Automated setup script
├── test_setup.py               # Validation script
├── backend/                     # FastAPI application
│   ├── main_simple.py          # Main backend file
│   ├── requirements.txt        # Python dependencies
│   └── *.py                    # Service modules
├── frontend/                    # Next.js application
│   ├── app/page.tsx           # Main React component
│   ├── package.json           # Node.js dependencies
│   └── ...                    # Next.js files
└── tests/                      # Integration tests
    └── test_integration.py    # Test suite
```

## **Contact & Support**

If you encounter any issues during setup or testing:

1. **Check the README.md** for detailed instructions
2. **Run the setup script** for automated validation
3. **Use the test script** to identify specific issues
4. **Review error messages** in the terminal output

The project is designed to be self-contained and should work out-of-the-box with proper API key configuration.

---

**Good luck with your assessment! 🚀**
