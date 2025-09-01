#!/bin/bash

# Setup environment for RAG Document Q&A System

echo "Setting up RAG Document Q&A environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please update .env file with your OpenAI API key"
fi

echo "Environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To run: python main_simple.py"
