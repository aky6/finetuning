#!/bin/bash

echo "🚀 Starting Ollama with Open WebUI..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start the services
echo "📦 Starting containers..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker ps | grep -q ollama && docker ps | grep -q open-webui; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🌐 Access your Ollama UI at: http://localhost:3001"
    echo "🔗 Ollama API at: http://localhost:11434"
    echo ""
    echo "📋 Next steps:"
    echo "1. Open http://localhost:3001 in your browser"
    echo "2. Create an account or login"
    echo "3. Pull a model (e.g., llama2:7b-chat)"
    echo "4. Start chatting!"
    echo ""
    echo "🛑 To stop services: ./scripts/stop.sh"
else
    echo "❌ Failed to start services. Check logs with: docker-compose logs"
fi 