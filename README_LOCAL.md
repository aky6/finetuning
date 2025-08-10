# Ollama Local Setup with Existing UIs

This setup uses existing, well-maintained UI solutions for Ollama instead of building from scratch.

## Available UI Options

### 1. Open WebUI (Recommended)
- **Features**: Full-featured, modern interface, RAG support, plugins
- **URL**: http://localhost:3000
- **Start**: `./scripts/start_openwebui.sh`

### 2. Ollama WebUI (Simple)
- **Features**: Clean, simple interface, easy to use
- **URL**: http://localhost:3000
- **Start**: `./scripts/start_simple.sh`

## Quick Start

### Prerequisites
- Docker installed and running
- At least 8GB RAM (16GB recommended)
- 10GB free disk space

### Step 1: Clone and Setup
```bash
git clone <your-repo>
cd ollama-local
chmod +x scripts/*.sh
```

### Step 2: Choose Your UI

**Option A: Open WebUI (Recommended)**
```bash
./scripts/start_openwebui.sh
```

**Option B: Simple WebUI**
```bash
./scripts/start_simple.sh
```

### Step 3: Access the UI
1. Open http://localhost:3000 in your browser
2. Create an account (Open WebUI) or start directly (Simple WebUI)
3. Pull your first model: `llama2:7b-chat`

### Step 4: Stop Services
```bash
./scripts/stop.sh
```

## Model Recommendations

### For Testing (Smaller models)
```bash
# Via API
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama2:7b-chat"}'

# Or via UI
# Go to Models tab and click "Pull Model"
```

### For Production (Larger models)
```bash
# 13B model (better quality, more RAM needed)
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama2:13b-chat"}'

# 70B model (best quality, lots of RAM needed)
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama2:70b-chat"}'
```

## API Usage

### Basic Chat
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b-chat",
    "prompt": "What is machine learning?",
    "stream": false
  }'
```

### List Models
```bash
curl http://localhost:11434/api/tags
```

### Using Python API Wrapper
```python
from app.api_wrapper import OllamaAPI

api = OllamaAPI()

# List models
models = api.list_models()
print(models)

# Generate text
result = api.generate("llama2:7b-chat", "Hello, how are you?")
print(result)
```

## Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker info

# Check logs
docker-compose logs

# Restart Docker Desktop
```

### Out of memory
```bash
# Use smaller models
ollama pull llama2:7b-chat

# Or increase Docker memory limit in Docker Desktop settings
```

### Model download fails
```bash
# Check internet connection
# Try again - downloads can be interrupted
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama2:7b-chat"}'
```

## Performance Tips

### For M1/M2 Macs
- Use CPU-only models for better performance
- Models run faster on Apple Silicon

### For Intel Macs
- Consider using smaller models
- Close other applications to free up RAM

### For Windows/Linux
- Ensure Docker has enough allocated memory
- Use SSD storage for better performance

## Next Steps

Once you're comfortable with the local setup:

1. **Test different models**: Try various Llama variants
2. **Experiment with parameters**: Temperature, top_p, etc.
3. **Try fine-tuning**: Use the pipeline scripts
4. **Move to AWS**: Use the Terraform setup for production

## Useful Commands

```bash
# View running containers
docker ps

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Clean up everything
docker-compose down -v
docker system prune -a
``` 