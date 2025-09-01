#!/bin/bash

echo "🚀 Starting RAG Document Q&A Project"
echo "======================================"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "Please run: export OPENAI_API_KEY='your_actual_api_key_here'"
    exit 1
fi

echo "✅ OpenAI API Key found"

# Start backend in background
echo "🔧 Starting backend on http://localhost:8001..."
cd backend
python main_simple.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "🎨 Starting frontend on http://localhost:3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 Project started successfully!"
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend:  http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
