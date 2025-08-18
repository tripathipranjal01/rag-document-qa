# 🔧 Environment Variables Configuration

## **Render Environment Variables**

### **Current Configuration (render.yaml):**

```yaml
envVars:
  - key: OPENAI_API_KEY
    sync: false  # Set manually in Render dashboard
  - key: PYTHON_VERSION
    value: 3.9.18
  - key: PORT
    value: 10000
  - key: MAX_UPLOAD_MB
    value: 30
  - key: ALLOWED_ORIGINS
    value: "*"
  - key: ENVIRONMENT
    value: production
```

---

## **Required Environment Variables:**

### **1. OPENAI_API_KEY** ⚠️ **REQUIRED**
- **Purpose**: OpenAI API key for embeddings and LLM
- **How to set**: 
  1. Go to Render dashboard
  2. Select your service
  3. Go to "Environment" tab
  4. Add: `OPENAI_API_KEY` = `sk-proj-...`
- **Format**: `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### **2. PYTHON_VERSION**
- **Purpose**: Python runtime version
- **Value**: `3.9.18`
- **Status**: ✅ Already configured

### **3. PORT**
- **Purpose**: Server port (Render sets this automatically)
- **Value**: `10000` (Render default)
- **Status**: ✅ Already configured

---

## **Optional Environment Variables:**

### **4. MAX_UPLOAD_MB**
- **Purpose**: Maximum file upload size
- **Value**: `30` (30MB limit)
- **Status**: ✅ Already configured

### **5. ALLOWED_ORIGINS**
- **Purpose**: CORS allowed origins
- **Value**: `"*"` (allows all origins)
- **Status**: ✅ Already configured

### **6. ENVIRONMENT**
- **Purpose**: Environment mode
- **Value**: `production`
- **Status**: ✅ Already configured

---

## **How to Set Environment Variables in Render:**

### **Method 1: Render Dashboard (Recommended)**
1. Go to [render.com](https://render.com)
2. Select your service: `rag-document-qa`
3. Click "Environment" tab
4. Add environment variables:
   ```
   OPENAI_API_KEY = sk-proj-your-actual-key-here
   ```

### **Method 2: render.yaml (Already Done)**
- Most variables are already configured in `render.yaml`
- Only `OPENAI_API_KEY` needs manual setting (for security)

---

## **Environment Variables in Code:**

### **Backend Usage:**
```python
# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# Port
port = int(os.getenv("PORT", 8001))

# Upload size limit
max_size = int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024

# CORS origins
allow_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
```

---

## **Frontend Environment Variables (Vercel):**

### **Required for Vercel:**
```bash
NEXT_PUBLIC_API_URL = https://rag-document-qa-fv4c.onrender.com
```

### **How to Set in Vercel:**
1. Go to Vercel dashboard
2. Select your project
3. Go to "Settings" → "Environment Variables"
4. Add: `NEXT_PUBLIC_API_URL` = `https://rag-document-qa-fv4c.onrender.com`

---

## **Testing Environment Variables:**

### **Check if API Key is Set:**
```bash
curl https://rag-document-qa-fv4c.onrender.com/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "api_key": {
    "status": "valid",
    "message": "API key is configured"
  }
}
```

### **If API Key is Missing:**
```json
{
  "status": "error",
  "api_key": {
    "status": "missing",
    "message": "API key not configured"
  }
}
```

---

## **Security Best Practices:**

### **✅ Do:**
- Set `OPENAI_API_KEY` manually in Render dashboard
- Use `sync: false` for sensitive variables
- Use production environment variables

### **❌ Don't:**
- Commit API keys to Git
- Use hardcoded values in code
- Share API keys publicly

---

## **Troubleshooting:**

### **Common Issues:**

1. **"API key not configured"**
   - Set `OPENAI_API_KEY` in Render dashboard

2. **"CORS errors"**
   - Check `ALLOWED_ORIGINS` is set to `"*"`

3. **"File upload too large"**
   - Increase `MAX_UPLOAD_MB` if needed

4. **"Port already in use"**
   - Render handles this automatically

---

## **Current Status:**
- ✅ **render.yaml**: Configured with all variables
- ⚠️ **OPENAI_API_KEY**: Needs manual setting in Render dashboard
- ✅ **All other variables**: Automatically set
- ✅ **Frontend variables**: Ready for Vercel deployment
