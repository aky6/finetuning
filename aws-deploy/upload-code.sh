#!/bin/bash
# Upload your local code to AWS EC2 instance

# Configuration - UPDATED WITH YOUR EC2
EC2_IP="18.175.154.125"              # Your EC2 Public IP
KEY_PATH="~/.ssh/ollama.pem"         # Your key file

echo "📦 Uploading code to AWS EC2..."

# Create finetuning directory on server
ssh -i $KEY_PATH ubuntu@$EC2_IP "mkdir -p /home/ubuntu/ollama-ai/finetuning"

# Upload finetuning code
echo "⬆️  Uploading fine-tuning pipeline..."
scp -i $KEY_PATH -r finetuning/* ubuntu@$EC2_IP:/home/ubuntu/ollama-ai/finetuning/

# Upload merge script
scp -i $KEY_PATH merge_and_export.py ubuntu@$EC2_IP:/home/ubuntu/ollama-ai/

# Setup remote environment
ssh -i $KEY_PATH ubuntu@$EC2_IP << 'EOF'
cd /home/ubuntu/ollama-ai

# Activate environment
source finetuning_env/bin/activate

# Install fine-tuning dependencies
cd finetuning
pip install -r requirements.txt

# Install additional packages
pip install openpyxl xlrd

# Create systemd service for fine-tuning UI
sudo tee /etc/systemd/system/finetuning-ui.service > /dev/null <<EOL
[Unit]
Description=Fine-tuning Web UI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ollama-ai/finetuning/web_interface
Environment=PATH=/home/ubuntu/ollama-ai/finetuning_env/bin
ExecStart=/home/ubuntu/ollama-ai/finetuning_env/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Enable and start fine-tuning service
sudo systemctl daemon-reload
sudo systemctl enable finetuning-ui
sudo systemctl start finetuning-ui

echo "✅ Fine-tuning setup complete!"
EOF

echo "🎉 Deployment complete!"
echo ""
echo "🌐 Open WebUI: http://$EC2_IP:3001"
echo "🔧 Fine-tuning UI: http://$EC2_IP:8888"
echo "📡 Ollama API: http://$EC2_IP:11434"