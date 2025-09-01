# Test file for chat history functionality

import requests
import json

def test_chat_history_all():
    """Test the all chat history endpoint"""
    try:
        response = requests.get("http://localhost:8001/api/chat-history/all")
        print(f"All chat history status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"All chat history test failed: {e}")
        return False

def test_chat_history_document():
    """Test the document-specific chat history endpoint"""
    try:
        # Test with a dummy document ID
        doc_id = "test-doc-123"
        response = requests.get(f"http://localhost:8001/api/chat-history/{doc_id}")
        print(f"Document chat history status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code in [200, 404]
    except Exception as e:
        print(f"Document chat history test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Chat History API...")
    
    all_history_ok = test_chat_history_all()
    doc_history_ok = test_chat_history_document()
    
    if all_history_ok and doc_history_ok:
        print("All chat history tests passed!")
    else:
        print("Some chat history tests failed!")
