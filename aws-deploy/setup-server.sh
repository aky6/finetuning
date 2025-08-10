#!/bin/bash
# AWS Server Setup Script - Run this on your EC2 instance

set -e

echo "🚀 Setting up Ollama AI Server on AWS..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies
sudo apt install -y curl wget git build-essential

# Install Ollama
echo "📦 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to start
sleep 10

# Pull models
echo "🦙 Pulling AI models..."
ollama pull tinyllama
ollama pull llama3.1:8b

# Create app directory
mkdir -p /home/ubuntu/ollama-ai
cd /home/ubuntu/ollama-ai

# Create Python virtual environment
python3.11 -m venv finetuning_env
source finetuning_env/bin/activate

# Install Open WebUI (matching your local native setup)
pip install open-webui

# Create systemd services (matching your local approach - fully native)
sudo tee /etc/systemd/system/ollama-webui.service > /dev/null <<EOF
[Unit]
Description=Ollama Web UI (Native)
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ollama-ai
Environment=PATH=/home/ubuntu/ollama-ai/finetuning_env/bin:/usr/local/bin:/usr/bin:/bin
Environment=OLLAMA_BASE_URL=http://localhost:11434
ExecStart=/home/ubuntu/ollama-ai/finetuning_env/bin/open-webui serve --host 0.0.0.0 --port 3001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Open WebUI service
sudo systemctl daemon-reload
sudo systemctl enable ollama-webui
sudo systemctl start ollama-webui

echo "✅ Basic setup complete!"
echo "🌐 Ollama API: http://YOUR_EC2_IP:11434"
echo "🌐 Open WebUI: http://YOUR_EC2_IP:3001"
echo ""
echo "📋 Next: Upload your fine-tuning code with upload-code.sh"