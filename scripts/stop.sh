#!/bin/bash

echo "🛑 Stopping Ollama services..."

# Stop all containers
docker-compose down
docker-compose -f docker-compose-simple.yml down

# Remove containers
docker rm -f ollama open-webui ollama-webui 2>/dev/null || true

echo "✅ Services stopped successfully!"
echo ""
echo "💡 To start again:"
echo "  - Open WebUI: ./scripts/start_openwebui.sh"
echo "  - Simple WebUI: ./scripts/start_simple.sh" 