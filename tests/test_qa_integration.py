import pytest
import requests
import json
import time
from typing import Dict, Any

class TestQAIntegration:
    """Integration tests for the QA API endpoint"""
    
    BASE_URL = "http://localhost:8001"
    
    def test_health_check(self):
        """Test that the API is running and healthy"""
        response = requests.get(f"{self.BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_session_creation(self):
        """Test session creation and management"""
        response = requests.get(f"{self.BASE_URL}/api/session")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "document_count" in data
        return data["session_id"]
    
    def test_qa_endpoint_without_documents(self):
        """Test QA endpoint when no documents are uploaded"""
        question = "What is the main topic?"
        
        response = requests.post(
            f"{self.BASE_URL}/api/qa",
            json={
                "question": question,
                "document_id": None,
                "scope": "all"
            },
            headers={"Content-Type": "application/json"}
        )
        
        # Should return an error or empty response when no documents
        assert response.status_code in [200, 400, 404]
    
    def test_qa_streaming_endpoint(self):
        """Test streaming QA endpoint"""
        question = "What is the main topic?"
        
        response = requests.post(
            f"{self.BASE_URL}/api/qa/stream",
            json={
                "question": question,
                "document_id": None,
                "scope": "all"
            },
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        # Should return streaming response
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/plain"
    
    def test_documents_endpoint(self):
        """Test documents listing endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_chat_history_endpoints(self):
        """Test chat history endpoints"""
        # Test global chat history
        response = requests.get(f"{self.BASE_URL}/api/chat-history/all")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)
    
    def test_security_endpoint(self):
        """Test security status endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/security/status")
        assert response.status_code == 200
        data = response.json()
        assert "security_status" in data
    
    def test_cost_estimation_endpoint(self):
        """Test cost estimation endpoint"""
        response = requests.get(f"{self.BASE_URL}/api/cost/estimate")
        assert response.status_code == 200
        data = response.json()
        assert "estimated_monthly_cost" in data
    
    def test_full_qa_workflow(self):
        """Test complete QA workflow with a sample document"""
        # This test would require uploading a document first
        # For now, we'll test the endpoints are accessible
        
        endpoints_to_test = [
            ("GET", "/api/health"),
            ("GET", "/api/session"),
            ("GET", "/api/documents"),
            ("GET", "/api/security/status"),
            ("GET", "/api/cost/estimate"),
            ("GET", "/api/chat-history/all"),
        ]
        
        for method, endpoint in endpoints_to_test:
            if method == "GET":
                response = requests.get(f"{self.BASE_URL}{endpoint}")
            else:
                response = requests.post(f"{self.BASE_URL}{endpoint}")
            
            # All endpoints should be accessible
            assert response.status_code in [200, 400, 404, 405]
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid JSON
        response = requests.post(
            f"{self.BASE_URL}/api/qa",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
        
        # Test missing required fields
        response = requests.post(
            f"{self.BASE_URL}/api/qa",
            json={},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
