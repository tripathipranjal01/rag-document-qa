# 🚀 Deployment Guide - Separate Frontend & Backend

## **Current Status:**
- ✅ **Backend**: Successfully deployed on Render
- 🔄 **Frontend**: Ready for Vercel deployment

## **Deployment Strategy:**
- **Backend**: Render (Python/FastAPI) - Already working
- **Frontend**: Vercel (Next.js) - To be deployed

---

## **Step 1: Backend (Already Deployed)**
**URL**: `https://rag-document-qa-fv4c.onrender.com`

**Status**: ✅ Working perfectly
- Health check: `/api/health`
- API docs: `/docs`
- All endpoints functional

---

## **Step 2: Deploy Frontend to Vercel**

### **Option A: Deploy via Vercel CLI**

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

3. **Deploy to Vercel:**
   ```bash
   vercel
   ```

4. **Follow the prompts:**
   - Link to existing project or create new
   - Set project name: `rag-document-qa-frontend`
   - Confirm deployment

### **Option B: Deploy via Vercel Dashboard**

1. **Go to [vercel.com](https://vercel.com)**
2. **Create new project**
3. **Import from GitHub:**
   - Repository: `tripathipranjal01/rag-document-qa`
   - Root directory: `frontend`
   - Framework preset: Next.js

4. **Set Environment Variables:**
   - `NEXT_PUBLIC_API_URL`: `https://rag-document-qa-fv4c.onrender.com`

5. **Deploy**

---

## **Step 3: Test the Complete System**

### **Frontend URL**: `https://your-frontend-url.vercel.app`

### **Test Commands:**
```bash
# Test backend health
curl https://rag-document-qa-fv4c.onrender.com/api/health

# Test frontend
curl https://your-frontend-url.vercel.app
```

---

## **Step 4: Update README with Final URLs**

Once deployed, update the README with:
- **Frontend URL**: `https://your-frontend-url.vercel.app`
- **Backend URL**: `https://rag-document-qa-fv4c.onrender.com`
- **API Documentation**: `https://rag-document-qa-fv4c.onrender.com/docs`

---

## **Benefits of This Approach:**

### **✅ Advantages:**
- **Best of both worlds**: Render for backend, Vercel for frontend
- **No root URL issues**: Frontend handles the main URL
- **Better performance**: Static frontend on Vercel
- **Automatic deployments**: Both platforms support auto-deploy
- **Scalability**: Each service can scale independently

### **🔧 Configuration:**
- **CORS**: Already configured for all origins
- **Environment variables**: Set in Vercel dashboard
- **API communication**: Frontend connects to Render backend

---

## **For Assessment Submission:**

### **Provide These URLs:**
1. **Main Application**: `https://your-frontend-url.vercel.app`
2. **Backend API**: `https://rag-document-qa-fv4c.onrender.com`
3. **API Documentation**: `https://rag-document-qa-fv4c.onrender.com/docs`

### **Demo Steps:**
1. Open frontend URL
2. Upload a document
3. Ask questions
4. Show document viewer
5. Demonstrate all features

---

## **Troubleshooting:**

### **If Frontend Can't Connect to Backend:**
1. Check CORS settings in backend
2. Verify environment variable `NEXT_PUBLIC_API_URL`
3. Test backend health endpoint

### **If Deployment Fails:**
1. Check build logs in Vercel
2. Verify all dependencies are installed
3. Check for TypeScript errors

---

## **Final Result:**
- **Complete RAG system** with beautiful UI
- **Professional deployment** on industry-standard platforms
- **Full functionality** for assessment demonstration
