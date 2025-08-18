#!/usr/bin/env python3
"""
Test script for RAG Document Q&A System
This script validates that everything is set up correctly.
"""

import requests
import time
import sys

def test_backend_health():
    """Test if backend is running and healthy"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get("http://localhost:8001/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is running")
            print(f"   Status: {data.get('status', 'unknown')}")
            
            # Check API key status
            api_key_info = data.get('api_key', {})
            if api_key_info.get('status') == 'valid':
                print("✅ API key is valid")
            else:
                print(f"❌ API key issue: {api_key_info.get('message', 'Unknown error')}")
                return False
            
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running")
        print("   Please start the backend: cd backend && python main_simple.py")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_frontend():
    """Test if frontend is accessible"""
    print("\n🔍 Testing frontend...")
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is running")
            return True
        else:
            print(f"❌ Frontend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Frontend is not running")
        print("   Please start the frontend: cd frontend && npm run dev")
        return False
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")
        return False

def test_upload_endpoint():
    """Test if upload endpoint is working"""
    print("\n🔍 Testing upload endpoint...")
    try:
        # Create a simple test file
        test_content = "This is a test document for validation."
        files = {'file': ('test.txt', test_content, 'text/plain')}
        
        response = requests.post("http://localhost:8001/api/upload", files=files, timeout=10)
        if response.status_code == 200:
            print("✅ Upload endpoint is working")
            return True
        else:
            print(f"❌ Upload endpoint returned status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing upload: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 RAG Document Q&A System - Setup Test")
    print("=" * 60)
    
    print("\n⚠️  Make sure both backend and frontend are running:")
    print("   Backend:  cd backend && python main_simple.py")
    print("   Frontend: cd frontend && npm run dev")
    print("\n   Press Enter when both are running...")
    input()
    
    # Test components
    backend_ok = test_backend_health()
    frontend_ok = test_frontend()
    upload_ok = test_upload_endpoint() if backend_ok else False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Backend: {'✅' if backend_ok else '❌'}")
    print(f"   Frontend: {'✅' if frontend_ok else '❌'}")
    print(f"   Upload: {'✅' if upload_ok else '❌'}")
    
    if all([backend_ok, frontend_ok, upload_ok]):
        print("\n🎉 All tests passed! Your setup is working correctly.")
        print("\n🚀 You can now:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Upload documents (PDF, DOCX, TXT, Images)")
        print("   3. Ask questions about your documents")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        if not backend_ok:
            print("   - Make sure backend is running on port 8001")
        if not frontend_ok:
            print("   - Make sure frontend is running on port 3000")
        if not upload_ok:
            print("   - Check API key configuration")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
