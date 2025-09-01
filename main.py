#!/usr/bin/env python3
"""
Main entry point for the RAG Document Q&A Backend
This file serves as the entry point for Render deployment
"""

import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run the main application
from main_advanced import app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting RAG Backend on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
