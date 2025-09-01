# Test file for progress tracking functionality

import requests
import json

def test_progress_endpoint():
    """Test the progress tracking endpoint"""
    try:
        # Test with a dummy document ID
        doc_id = "test-doc-123"
        response = requests.get(f"http://localhost:8001/api/progress/{doc_id}")
        print(f"Progress endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Progress endpoint test failed: {e}")
        return False

def test_security_status():
    """Test the security status endpoint"""
    try:
        response = requests.get("http://localhost:8001/api/security/status")
        print(f"Security status status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Security status test failed: {e}")
        return False

def test_cost_estimate():
    """Test the cost estimation endpoint"""
    try:
        response = requests.get("http://localhost:8001/api/cost/estimate")
        print(f"Cost estimate status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Cost estimate test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Progress and Monitoring API...")
    
    progress_ok = test_progress_endpoint()
    security_ok = test_security_status()
    cost_ok = test_cost_estimate()
    
    if progress_ok and security_ok and cost_ok:
        print("All progress and monitoring tests passed!")
    else:
        print("Some progress and monitoring tests failed!")
