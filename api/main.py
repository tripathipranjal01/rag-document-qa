from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import your main FastAPI app
from main_simple import app

# This is the entry point for Vercel
# The app is already configured in main_simple.py
handler = app
