# Minimal version of the RAG Document Q&A system
# Basic functionality only

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Minimal Document Q&A API",
    description="Basic AI-powered document analysis and Q&A system",
    version="1.0.0"
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
    return {"message": "Minimal RAG Document Q&A API is running!"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
