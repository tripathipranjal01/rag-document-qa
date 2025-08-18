# Deployment Guide

## **Option 1: Railway (Recommended - Free & Easy)**

### **Step 1: Create Railway Account**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Create a new project

### **Step 2: Deploy Backend**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

### **Step 3: Set Environment Variables**
In Railway dashboard:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PORT`: 8001 (auto-set by Railway)

### **Step 4: Get Your URL**
Railway will provide: `https://your-app-name.railway.app`

## **Option 2: Heroku**

### **Step 1: Create Heroku Account**
1. Go to [heroku.com](https://heroku.com)
2. Create account and install CLI

### **Step 2: Deploy**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-rag-app

# Set environment variables
heroku config:set OPENAI_API_KEY="your_api_key_here"

# Deploy
git push heroku main
```

### **Step 3: Get Your URL**
Heroku will provide: `https://your-rag-app.herokuapp.com`

## **Option 3: Render**

### **Step 1: Create Render Account**
1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### **Step 2: Deploy**
1. Connect your GitHub repository
2. Create new Web Service
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `cd backend && python main_simple.py`
5. Add environment variable: `OPENAI_API_KEY`

## **Testing Deployment**

### **1. Health Check**
```bash
curl https://your-app-url.railway.app/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "Advanced RAG API is running",
  "api_key": {
    "status": "valid",
    "message": "API key is configured"
  }
}
```

### **2. Run Integration Tests**
```bash
# Update the URL in tests/test_integration.py
python tests/test_integration.py
```

### **3. Manual Testing**
1. Open your deployment URL
2. Upload a test document
3. Ask questions
4. Verify citations work

## **Environment Variables**

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `PORT` | Port number (auto-set by platform) | No |

## **Troubleshooting**

### **Common Issues**

**"Application Error"**
- Check logs: `railway logs` or `heroku logs --tail`
- Verify environment variables are set
- Ensure all dependencies are in requirements.txt

**"API Key Error"**
- Verify OPENAI_API_KEY is set correctly
- Check API key has sufficient credits
- Test API key manually

**"Port Issues"**
- Ensure app listens on `0.0.0.0:PORT`
- Check platform-specific port configuration

### **Logs and Monitoring**

**Railway:**
```bash
railway logs
railway status
```

**Heroku:**
```bash
heroku logs --tail
heroku ps
```

## **Cost Estimation**

### **Railway (Free Tier)**
- 500 hours/month free
- $5/month for additional usage
- Perfect for demos and testing

### **Heroku (Free Tier Discontinued)**
- Basic dyno: $7/month
- Eco dyno: $5/month

### **Render (Free Tier)**
- 750 hours/month free
- $7/month for always-on service

## **Security Checklist**

- ✅ No secrets in repository
- ✅ Environment variables for sensitive data
- ✅ CORS properly configured
- ✅ Input validation implemented
- ✅ Error handling without sensitive info

## **Performance Optimization**

### **For Production**
1. Add Redis for caching
2. Use PostgreSQL with pgvector
3. Implement rate limiting
4. Add CDN for static files
5. Monitor with APM tools

### **Current Setup**
- In-memory storage (good for demos)
- Local file persistence
- Basic caching implementation
- Suitable for small to medium scale
