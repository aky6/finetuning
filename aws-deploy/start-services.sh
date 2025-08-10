#!/bin/bash
# Start all services on AWS (matches your local native setup)

echo "🚀 Starting Ollama AI Services on AWS..."

# Check if we're on the EC2 instance
if [ ! -d "/home/ubuntu/ollama-ai" ]; then
    echo "❌ This script should be run on the EC2 instance"
    echo "📋 Usage: ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_IP './start-services.sh'"
    exit 1
fi

cd /home/ubuntu/ollama-ai

# Start Ollama service
echo "🦙 Starting Ollama service..."
sudo systemctl start ollama
sudo systemctl enable ollama

# Start Open WebUI (native)
echo "🌐 Starting Open WebUI (native)..."
sudo systemctl start ollama-webui
sudo systemctl enable ollama-webui

# Start Fine-tuning Web Interface
echo "🔧 Starting Fine-tuning Web Interface..."
sudo systemctl start finetuning-ui
sudo systemctl enable finetuning-ui

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Checking service status..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if systemctl is-active --quiet ollama; then
    echo "✅ Ollama: Running"
else
    echo "❌ Ollama: Failed"
fi

if systemctl is-active --quiet ollama-webui; then
    echo "✅ Open WebUI: Running"
else
    echo "❌ Open WebUI: Failed"
fi

if systemctl is-active --quiet finetuning-ui; then
    echo "✅ Fine-tuning UI: Running"
else
    echo "❌ Fine-tuning UI: Failed"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

echo ""
echo "🎉 Services started! Access your AI:"
echo "🌐 Open WebUI: http://$PUBLIC_IP:3001"
echo "🔧 Fine-tuning UI: http://$PUBLIC_IP:8888"
echo "📡 Ollama API: http://$PUBLIC_IP:11434"
echo ""
echo "📋 Management commands:"
echo "  • Check status: sudo systemctl status ollama-webui"
echo "  • View logs: sudo journalctl -u ollama-webui -f"
echo "  • Restart: sudo systemctl restart ollama-webui"
echo ""