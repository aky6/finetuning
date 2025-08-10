#!/bin/bash
# Audit your local setup before AWS deployment

echo "🔍 Auditing Local Setup..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check what's running
echo "📊 Current Running Services:"
echo ""

# Check Ollama
if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama: Running (native)"
    echo "   📡 API: http://localhost:11434"
else
    echo "❌ Ollama: Not running"
fi

# Check Open WebUI
if lsof -i :3001 > /dev/null 2>&1; then
    echo "✅ Open WebUI: Running on port 3001"
    echo "   🌐 URL: http://localhost:3001"
else
    echo "❌ Open WebUI: Not running on port 3001"
fi

# Check Fine-tuning UI
if lsof -i :8888 > /dev/null 2>&1; then
    echo "✅ Fine-tuning UI: Running on port 8888"
    echo "   🔧 URL: http://localhost:8888"
else
    echo "❌ Fine-tuning UI: Not running on port 8888"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Files to Deploy:"
echo ""

# Check important files
if [ -f "finetuning/requirements.txt" ]; then
    echo "✅ finetuning/requirements.txt"
else
    echo "❌ finetuning/requirements.txt"
fi

if [ -f "merge_and_export.py" ]; then
    echo "✅ merge_and_export.py"
else
    echo "❌ merge_and_export.py"
fi

if [ -d "app" ]; then
    echo "✅ app/ directory"
else
    echo "❌ app/ directory"
fi

if [ -d "scripts" ]; then
    echo "✅ scripts/ directory"
else
    echo "❌ scripts/ directory"
fi

if [ -d "finetuning/web_interface" ]; then
    echo "✅ finetuning/web_interface"
else
    echo "❌ finetuning/web_interface"
fi

if [ -d "finetuning_env" ]; then
    echo "✅ finetuning_env (won't be transferred - will be recreated)"
else
    echo "❌ finetuning_env"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🦙 Available Models:"
echo ""
ollama list

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Ready for AWS Deployment!"
echo ""
echo "📋 Next Steps:"
echo "1. Create EC2 instance (m5.2xlarge, Ubuntu 22.04)"
echo "2. Upload and run: aws-deploy/setup-server.sh"
echo "3. Edit aws-deploy/upload-code.sh with your EC2 IP"
echo "4. Run: aws-deploy/upload-code.sh"
echo "5. SSH and run: ./start-services.sh"
echo ""