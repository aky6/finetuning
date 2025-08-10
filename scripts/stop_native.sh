#!/bin/bash

echo "🛑 Stopping Native Ollama setup..."

# Stop the web UI container
echo "📦 Stopping Open WebUI container..."
docker-compose -f docker-compose-native.yml down

# Ask if user wants to stop native Ollama too
echo ""
read -p "Do you want to stop native Ollama as well? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 Stopping native Ollama..."
    pkill -f "ollama serve"
    echo "✅ Native Ollama stopped"
else
    echo "✅ Native Ollama left running"
fi

echo ""
echo "✅ Web UI stopped successfully!"
echo ""
echo "💡 To start again: ./scripts/start_native.sh"