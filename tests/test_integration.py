import pytest
import requests
import json
import time
import os
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8001"
API_BASE = f"{BASE_URL}/api"

class TestRAGIntegration:
    """Integration tests for the RAG Document Q&A system"""
    
    def setup_method(self):
        """Setup before each test"""
        self.session = requests.Session()
        # Get a session ID by making a request
        response = self.session.get(f"{API_BASE}/session")
        assert response.status_code == 200
        self.session_data = response.json()
        self.session_id = self.session_data.get("session_id")
        assert self.session_id is not None
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    def test_session_management(self):
        """Test session creation and management"""
        # Test session creation
        response = self.session.get(f"{API_BASE}/session")
        assert response.status_code == 200
        session_data = response.json()
        assert "session_id" in session_data
        assert "document_count" in session_data
        
        # Test session deletion
        response = self.session.delete(f"{API_BASE}/session")
        assert response.status_code == 200
        
        # Verify session is cleared
        response = self.session.get(f"{API_BASE}/documents")
        assert response.status_code == 200
        documents = response.json()
        assert len(documents) == 0
    
    def test_document_upload_and_qa(self):
        """Test complete flow: upload document and ask questions"""
        # Create a test document
        test_content = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
        Machine Learning is a subset of AI that enables computers to learn and improve from experience.
        Deep Learning is a subset of machine learning that uses neural networks with multiple layers.
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.
        """
        
        # Create test file
        test_file_path = "test_document.txt"
        with open(test_file_path, "w") as f:
            f.write(test_content)
        
        try:
            # Upload document
            with open(test_file_path, "rb") as f:
                files = {"file": ("test_document.txt", f, "text/plain")}
                response = self.session.post(f"{API_BASE}/upload", files=files)
            
            assert response.status_code == 200
            upload_data = response.json()
            assert "filename" in upload_data
            assert upload_data["filename"] == "test_document.txt"
            
            # Wait for processing
            time.sleep(2)
            
            # Check document status
            response = self.session.get(f"{API_BASE}/documents")
            assert response.status_code == 200
            documents = response.json()
            assert len(documents) == 1
            assert documents[0]["filename"] == "test_document.txt"
            assert documents[0]["status"] == "indexed"
            
            # Ask a question
            question_data = {
                "question": "What is Artificial Intelligence?",
                "document_id": None,
                "scope": "all"
            }
            
            response = self.session.post(
                f"{API_BASE}/qa",
                json=question_data,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            qa_data = response.json()
            
            # Verify response structure
            assert "answer" in qa_data
            assert "citations" in qa_data
            assert "sources" in qa_data
            assert "processing_time" in qa_data
            assert "chunks_used" in qa_data
            
            # Verify answer quality
            answer = qa_data["answer"].lower()
            assert "artificial intelligence" in answer or "ai" in answer
            assert len(qa_data["citations"]) > 0
            
            # Test document-specific question
            doc_id = documents[0]["id"]
            question_data = {
                "question": "What is Machine Learning?",
                "document_id": doc_id,
                "scope": "document"
            }
            
            response = self.session.post(
                f"{API_BASE}/qa",
                json=question_data,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            qa_data = response.json()
            assert "machine learning" in qa_data["answer"].lower()
            
        finally:
            # Cleanup
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
    
    def test_document_viewer(self):
        """Test document viewer functionality"""
        # First upload a document
        test_content = "This is a test document for the viewer functionality."
        test_file_path = "test_viewer.txt"
        
        with open(test_file_path, "w") as f:
            f.write(test_content)
        
        try:
            # Upload document
            with open(test_file_path, "rb") as f:
                files = {"file": ("test_viewer.txt", f, "text/plain")}
                response = self.session.post(f"{API_BASE}/upload", files=files)
            
            assert response.status_code == 200
            
            # Wait for processing
            time.sleep(2)
            
            # Get documents
            response = self.session.get(f"{API_BASE}/documents")
            assert response.status_code == 200
            documents = response.json()
            assert len(documents) > 0
            
            doc_id = documents[0]["id"]
            
            # Test document content endpoint
            response = self.session.get(f"{API_BASE}/documents/{doc_id}/content")
            assert response.status_code == 200
            content_data = response.json()
            
            assert "id" in content_data
            assert "filename" in content_data
            assert "content" in content_data
            assert "chunks" in content_data
            assert "status" in content_data
            
            # Verify content
            assert "test document" in content_data["content"].lower()
            assert len(content_data["chunks"]) > 0
            
            # Test document status endpoint
            response = self.session.get(f"{API_BASE}/documents/{doc_id}/status")
            assert response.status_code == 200
            status_data = response.json()
            assert "status" in status_data
            
        finally:
            # Cleanup
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        # Test invalid file upload
        response = self.session.post(f"{API_BASE}/upload", files={"file": ("test.invalid", b"invalid content", "text/plain")})
        assert response.status_code == 400
        
        # Test Q&A without documents
        question_data = {
            "question": "What is AI?",
            "document_id": None,
            "scope": "all"
        }
        
        response = self.session.post(
            f"{API_BASE}/qa",
            json=question_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "detail" in error_data
        
        # Test invalid document ID
        response = self.session.get(f"{API_BASE}/documents/invalid-id/content")
        assert response.status_code == 404
    
    def test_citation_functionality(self):
        """Test citation and source tracking"""
        # Upload a document with specific content
        test_content = """
        Python is a high-level programming language.
        Python was created by Guido van Rossum in 1991.
        Python is known for its simplicity and readability.
        Python is widely used in data science and machine learning.
        """
        
        test_file_path = "test_citations.txt"
        with open(test_file_path, "w") as f:
            f.write(test_content)
        
        try:
            # Upload document
            with open(test_file_path, "rb") as f:
                files = {"file": ("test_citations.txt", f, "text/plain")}
                response = self.session.post(f"{API_BASE}/upload", files=files)
            
            assert response.status_code == 200
            
            # Wait for processing
            time.sleep(2)
            
            # Ask a question that should generate citations
            question_data = {
                "question": "Who created Python and when?",
                "document_id": None,
                "scope": "all"
            }
            
            response = self.session.post(
                f"{API_BASE}/qa",
                json=question_data,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            qa_data = response.json()
            
            # Verify citations
            assert len(qa_data["citations"]) > 0
            
            citation = qa_data["citations"][0]
            assert "text" in citation
            assert "page" in citation
            assert "source" in citation
            assert "similarity" in citation
            assert "chunk_id" in citation
            
            # Verify citation content
            citation_text = citation["text"].lower()
            assert "guido van rossum" in citation_text or "1991" in citation_text
            
        finally:
            # Cleanup
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
    
    def test_performance_metrics(self):
        """Test performance and response time metrics"""
        # Upload a document
        test_content = "Performance test document with multiple sentences. " * 50
        test_file_path = "test_performance.txt"
        
        with open(test_file_path, "w") as f:
            f.write(test_content)
        
        try:
            # Upload document
            with open(test_file_path, "rb") as f:
                files = {"file": ("test_performance.txt", f, "text/plain")}
                response = self.session.post(f"{API_BASE}/upload", files=files)
            
            assert response.status_code == 200
            
            # Wait for processing
            time.sleep(3)
            
            # Test response time
            start_time = time.time()
            question_data = {
                "question": "What is this document about?",
                "document_id": None,
                "scope": "all"
            }
            
            response = self.session.post(
                f"{API_BASE}/qa",
                json=question_data,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            assert response.status_code == 200
            qa_data = response.json()
            
            # Verify performance metrics
            assert "processing_time" in qa_data
            assert qa_data["processing_time"] > 0
            assert qa_data["processing_time"] < 30  # Should complete within 30 seconds
            
            # Verify response time is reasonable
            assert response_time < 35  # Total time including network overhead
            
        finally:
            # Cleanup
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
