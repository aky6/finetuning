# 🚀 **EC2 Instance Setup Guide**

## **📋 Step-by-Step EC2 Creation**

### **1. Create Key Pair**
```
AWS Console → EC2 → Key Pairs → Create key pair
- Name: ollama-key
- Type: RSA
- Format: .pem
- Download and save securely
```

### **2. Launch Instance**
```
AWS Console → EC2 → Launch instances

Basic Settings:
- Name: ollama-ai-server
- OS: Ubuntu Server 22.04 LTS
- Instance Type: m5.2xlarge (8 vCPU, 32 GB RAM)
- Key Pair: ollama-key

Network Settings:
- VPC: Default
- Public IP: Enable
- Security Group: Create new

Security Group Rules:
- SSH (22): Your IP only
- HTTP (80): 0.0.0.0/0
- Custom TCP (3001): 0.0.0/0 (Open WebUI)
- Custom TCP (8888): 0.0.0.0/0 (Fine-tuning UI)
- Custom TCP (11434): 0.0.0/0 (Ollama API)

Storage:
- Size: 100 GB (GP3)
- Delete on termination: No
```

### **3. Get Instance Details**
```
After launch, note:
- Public IP Address
- Instance ID
- Status: Running
```

### **4. Update Your Scripts**
```bash
# Edit aws-deploy/upload-code.sh
EC2_IP="YOUR_ACTUAL_PUBLIC_IP"
KEY_PATH="~/.ssh/ollama-key.pem"

# Move your key file
mv ~/Downloads/ollama-key.pem ~/.ssh/
chmod 400 ~/.ssh/ollama-key.pem
```

### **5. Deploy to AWS**
```bash
# Setup server
scp -i ~/.ssh/ollama-key.pem aws-deploy/setup-server.sh ubuntu@YOUR_IP:/home/ubuntu/
ssh -i ~/.ssh/ollama-key.pem ubuntu@YOUR_IP
sudo chmod +x setup-server.sh
./setup-server.sh

# Upload your code
./aws-deploy/upload-code.sh

# Start services
./aws-deploy/start-services.sh
```

## **💰 Estimated Costs**
- **m5.2xlarge**: ~$0.384/hour = ~$280/month
- **Storage**: ~$10/month (100 GB GP3)
- **Data Transfer**: ~$5-20/month
- **Total**: ~$300-350/month

## **🔒 Security Notes**
- Key pair is your only access - keep it secure
- Consider restricting SSH access to your IP only
- Monitor CloudWatch for unusual activity
- Regular security updates via Ubuntu

## **📱 Access URLs**
After deployment:
- **Open WebUI**: http://YOUR_IP:3001
- **Fine-tuning UI**: http://YOUR_IP:8888  
- **Ollama API**: http://YOUR_IP:11434 