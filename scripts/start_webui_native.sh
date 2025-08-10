#!/bin/bash

echo "🚀 Starting Native Ollama with Open WebUI (Python 3.11)..."

# Check if Ollama is running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "📦 Starting native Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    echo "Ollama started with PID: $OLLAMA_PID"
    sleep 3
else
    echo "✅ Ollama is already running"
fi

# Activate virtual environment and start Open WebUI
echo "🌐 Starting Open WebUI (Python 3.11)..."
cd "$(dirname "$0")/.."
source venv311/bin/activate

# Set environment variables
export OLLAMA_BASE_URL="http://localhost:11434"
export WEBUI_SECRET_KEY="your-secret-key-here"
export DEFAULT_USER_ROLE="admin"

# Start Open WebUI
echo ""
echo "✅ Starting Open WebUI..."
echo "🌐 Access your UI at: http://localhost:8000"
echo "🔗 Native Ollama API at: http://localhost:11434"
echo ""
echo "💡 Performance benefits:"
echo "  - Native Ollama with Apple M4 GPU acceleration"
echo "  - No Docker overhead"
echo "  - Python 3.11 virtual environment"
echo "  - Direct hardware access"
echo ""
echo "🛑 Press Ctrl+C to stop"
echo ""

open-webui serve --host 0.0.0.0 --port 8080