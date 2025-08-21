#!/usr/bin/env python3
"""
Simple test script to verify RAG functionality
"""
import os
from dotenv import load_dotenv
from rag_service import RAGService

def test_rag():
    load_dotenv()
    
    print("Testing RAG Service...")
    
    # Initialize RAG service
    rag = RAGService()
    
    if not rag.client:
        print("❌ OpenAI client not initialized")
        return False
    
    print("✅ OpenAI client initialized")
    
    # Test embedding creation
    try:
        embedding = rag.create_embedding("This is a test document about artificial intelligence.")
        print(f"✅ Embedding created successfully (length: {len(embedding)})")
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        return False
    
    # Test answer generation
    try:
        context = [
            {"text": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines."},
            {"text": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."}
        ]
        answer = rag.generate_answer("What is artificial intelligence?", context)
        print(f"✅ Answer generation successful: {answer[:100]}...")
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return False
    
    print("🎉 All RAG tests passed!")
    return True

if __name__ == "__main__":
    test_rag()
