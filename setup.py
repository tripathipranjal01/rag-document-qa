#!/usr/bin/env python3
"""
Setup script for RAG Document Q&A System
This script helps you quickly set up the project and validate your configuration.
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 60)
    print("🚀 RAG Document Q&A System - Setup Script")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n📋 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Error: Python 3.9+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_node_version():
    """Check if Node.js is installed and version is compatible"""
    print("\n📋 Checking Node.js version...")
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Node.js {version} - OK")
            return True
        else:
            print("❌ Node.js not found or error occurred")
            return False
    except FileNotFoundError:
        print("❌ Node.js not installed")
        print("   Please install Node.js 18+ from https://nodejs.org/")
        return False

def check_api_key():
    """Check if OpenAI API key is configured"""
    print("\n📋 Checking OpenAI API key...")
    
    # Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        print("✅ OpenAI API key found in environment variable")
        return True
    
    # Check main_simple.py file
    try:
        with open("backend/main_simple.py", "r") as f:
            content = f.read()
            if "your_openai_api_key_here" in content:
                print("❌ OpenAI API key not configured")
                print("   Please set your API key in one of these ways:")
                print("   1. Set environment variable: export OPENAI_API_KEY='your_key_here'")
                print("   2. Edit backend/main_simple.py and replace 'your_openai_api_key_here'")
                return False
            else:
                print("✅ OpenAI API key found in main_simple.py")
                return True
    except FileNotFoundError:
        print("❌ main_simple.py not found")
        return False

def install_backend_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing backend dependencies...")
    try:
        # Create virtual environment if it doesn't exist
        if not os.path.exists("backend/venv"):
            print("   Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "backend/venv"], check=True)
        
        # Activate virtual environment and install dependencies
        if platform.system() == "Windows":
            pip_path = "backend/venv/Scripts/pip"
        else:
            pip_path = "backend/venv/bin/pip"
        
        subprocess.run([pip_path, "install", "-r", "backend/requirements.txt"], check=True)
        print("✅ Backend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing backend dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install Node.js dependencies"""
    print("\n📦 Installing frontend dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd="frontend", check=True)
        print("✅ Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing frontend dependencies: {e}")
        return False

def main():
    print_header()
    
    # Check prerequisites
    python_ok = check_python_version()
    node_ok = check_node_version()
    api_key_ok = check_api_key()
    
    if not python_ok or not node_ok:
        print("\n❌ Prerequisites not met. Please install required software.")
        sys.exit(1)
    
    # Install dependencies
    backend_ok = install_backend_dependencies()
    frontend_ok = install_frontend_dependencies()
    
    if not backend_ok or not frontend_ok:
        print("\n❌ Failed to install dependencies.")
        sys.exit(1)
    
    # Final status
    print("\n" + "=" * 60)
    print("📊 Setup Summary:")
    print(f"   Python: {'✅' if python_ok else '❌'}")
    print(f"   Node.js: {'✅' if node_ok else '❌'}")
    print(f"   API Key: {'✅' if api_key_ok else '❌'}")
    print(f"   Backend Dependencies: {'✅' if backend_ok else '❌'}")
    print(f"   Frontend Dependencies: {'✅' if frontend_ok else '❌'}")
    
    if all([python_ok, node_ok, api_key_ok, backend_ok, frontend_ok]):
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 To run the application:")
        print("   1. Start backend: cd backend && python main_simple.py")
        print("   2. Start frontend: cd frontend && npm run dev")
        print("   3. Open http://localhost:3000 in your browser")
    else:
        print("\n⚠️  Setup completed with warnings.")
        if not api_key_ok:
            print("   Please configure your OpenAI API key before running the application.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
