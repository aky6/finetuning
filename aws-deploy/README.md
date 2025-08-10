# 🚀 Simple AWS Deployment Guide

Deploy your Ollama + Fine-tuning setup to AWS EC2 (No Docker, No Terraform)

## 📋 Prerequisites
- AWS Account
- SSH Key Pair created in EC2
- Your local setup working

## 🎯 Quick Deployment

### Step 1: Create EC2 Instance
1. Go to AWS EC2 Console
2. Launch Instance:
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Type**: `m5.2xlarge` (8 vCPU, 32GB RAM)
   - **Storage**: 50GB gp3
   - **Security Group**: Allow ports 22, 80, 3001, 8888, 11434

### Step 2: Connect & Setup Server
```bash
# Connect to your instance
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_IP

# Upload and run setup script
scp -i ~/.ssh/your-key.pem aws-deploy/setup-server.sh ubuntu@YOUR_EC2_IP:~
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_IP 'chmod +x setup-server.sh && ./setup-server.sh'
```

### Step 3: Upload Your Code
```bash
# Edit upload-code.sh with your EC2 IP and key path
nano aws-deploy/upload-code.sh

# Run upload script
chmod +x aws-deploy/upload-code.sh
./aws-deploy/upload-code.sh
```

## 🌐 Access Your Services

After deployment:
- **Open WebUI**: http://YOUR_EC2_IP:3001
- **Fine-tuning UI**: http://YOUR_EC2_IP:8888
- **Ollama API**: http://YOUR_EC2_IP:11434

## 🔧 Management Commands

```bash
# Check service status
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_IP
sudo systemctl status ollama
sudo systemctl status ollama-webui
sudo systemctl status finetuning-ui

# View logs
sudo journalctl -u ollama-webui -f
sudo journalctl -u finetuning-ui -f

# Restart services
sudo systemctl restart ollama-webui
sudo systemctl restart finetuning-ui
```

## 💰 Cost Estimation
- **m5.2xlarge**: ~$0.384/hour (~$276/month)
- **Storage (50GB)**: ~$5/month
- **Data Transfer**: Varies

## 🛡️ Security Notes
- Change default ports if needed
- Restrict security group to your IP for production
- Use HTTPS with SSL certificates for production
- Enable AWS CloudWatch for monitoring

## 📊 Monitoring
- AWS CloudWatch automatically enabled
- Access logs via EC2 console
- Monitor CPU/Memory usage
- Set up billing alerts

## 🔄 Backup Strategy
- AMI snapshots of your instance
- S3 backup for trained models
- Export fine-tuned models regularly