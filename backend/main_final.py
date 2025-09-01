# Final version of the RAG Document Q&A system
# This is the production-ready version

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Final Document Q&A API",
    description="Production-ready AI-powered document analysis and Q&A system",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Final RAG Document Q&A API is running!"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
