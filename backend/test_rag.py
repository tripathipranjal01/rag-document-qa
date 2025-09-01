# Test file for RAG functionality

import requests
import json

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8001/api/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_qa_endpoint():
    """Test the QA endpoint"""
    try:
        data = {
            "question": "What is the main topic?",
            "document_id": None,
            "scope": "all"
        }
        response = requests.post(
            "http://localhost:8001/api/qa",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"QA endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code in [200, 400, 404]
    except Exception as e:
        print(f"QA endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG API...")
    
    health_ok = test_health_check()
    qa_ok = test_qa_endpoint()
    
    if health_ok and qa_ok:
        print("All tests passed!")
    else:
        print("Some tests failed!")
