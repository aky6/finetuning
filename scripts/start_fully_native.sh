#!/bin/bash

echo "🚀 Starting Fully Native Ollama Setup..."

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

# Check if Open WebUI is installed
if ! command -v open-webui &> /dev/null; then
    echo "📦 Installing Open WebUI..."
    pip install open-webui
fi

# Set environment variables
export OLLAMA_BASE_URL="http://localhost:11434"

# Start Open WebUI
echo "🌐 Starting Open WebUI natively..."
echo "Access your UI at: http://localhost:8080"
echo "Press Ctrl+C to stop"

open-webui serve --host 0.0.0.0 --port 8080