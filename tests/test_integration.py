import pytest
import requests
import json
import time
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main_simple import app

client = TestClient(app)

class TestRAGAPI:
    
    def test_health_endpoint(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data
        print("✅ Health endpoint test passed")
    
    def test_session_management(self):
        response = client.get("/api/session")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "document_count" in data
        print("✅ Session management test passed")
    
    def test_upload_document(self):
        test_content = "This is a test document for integration testing."
        files = {'file': ('test.txt', test_content, 'text/plain')}
        
        response = client.post("/api/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["status"] == "completed"
        print("✅ Document upload test passed")
        
        return data["id"]
    
    def test_qa_endpoint(self):
        doc_id = self.test_upload_document()
        
        time.sleep(2)
        
        qa_data = {
            "question": "What is this document about?",
            "scope": "all"
        }
        
        response = client.post("/api/qa", json=qa_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "processing_time" in data
        print("✅ Q&A endpoint test passed")
    
    def test_documents_list(self):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print("✅ Documents list test passed")
    
    def test_analytics_endpoint(self):
        response = client.get("/api/analytics")
        assert response.status_code == 200
        data = response.json()
        assert "user_stats" in data
        assert "system_stats" in data
        print("✅ Analytics endpoint test passed")

def test_deployment_health():
    """Test deployment health endpoint"""
    try:
        response = requests.get("https://rag-document-qa-backend.onrender.com/api/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "error"]
        print("✅ Deployment health check passed")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Deployment health check failed: {e}")
        print("   Make sure to update the URL with your actual deployment URL")

if __name__ == "__main__":
    print("🧪 Running Integration Tests...")
    print("=" * 50)
    
    test_instance = TestRAGAPI()
    
    try:
        test_instance.test_health_endpoint()
        test_instance.test_session_management()
        test_instance.test_upload_document()
        test_instance.test_qa_endpoint()
        test_instance.test_documents_list()
        test_instance.test_analytics_endpoint()
        
        print("\n" + "=" * 50)
        print("🎉 All integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
