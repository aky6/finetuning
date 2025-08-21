

#!/bin/bash

echo "🚀 Starting Native Ollama with Web UI..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start the web UI container
echo "📦 Starting Open WebUI container..."
docker-compose -f docker-compose-native.yml up -d

# Wait for service to be ready
echo "⏳ Waiting for services to start..."
sleep 5

# Check if services are running
if docker ps | grep -q open-webui-native; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🌐 Access your Ollama UI at: http://localhost:3001"
    echo "🔗 Native Ollama API at: http://localhost:11434"
    echo ""
    echo "💡 Benefits of this setup:"
    echo "  - Much faster performance (native Ollama)"
    echo "  - Lower memory usage"
    echo "  - Direct hardware access"
    echo "  - Still get the great web UI"
    echo ""
    echo "📋 Next steps:"
    echo "1. Open http://localhost:3001 in your browser"
    echo "2. Create an account or login"
    echo "3. Your models should already be available!"
    echo ""
    echo "🛑 To stop: ./scripts/stop_native.sh"
else
    echo "❌ Failed to start web UI. Check logs with: docker-compose -f docker-compose-native.yml logs"
fi