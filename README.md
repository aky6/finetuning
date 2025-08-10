# Ollama Production Hosting & Fine-tuning Pipeline

A production-grade setup for hosting Ollama with Llama models on AWS, including a comprehensive fine-tuning pipeline.

## Architecture

- **Infrastructure**: AWS ECS with Fargate for containerized Ollama instances
- **Load Balancer**: Application Load Balancer with auto-scaling
- **Database**: RDS PostgreSQL for model metadata and training logs
- **Storage**: S3 for model artifacts and training data
- **Fine-tuning**: SageMaker for distributed training
- **Monitoring**: CloudWatch, Prometheus, and Grafana
- **CI/CD**: GitHub Actions with Terraform

## Quick Start

1. **Prerequisites**:
   - AWS CLI configured
   - Terraform installed
   - Docker installed
   - Python 3.9+

2. **Deploy Infrastructure**:
   ```bash
   cd terraform
   terraform init
   terraform plan
   terraform apply
   ```

3. **Deploy Application**:
   ```bash
   ./scripts/deploy.sh
   ```

4. **Access the UI**:
   - Web UI: `http://your-alb-domain`
   - API: `http://your-alb-domain/api`

## Features

- ✅ Multi-model support with automatic scaling
- ✅ Production-grade monitoring and logging
- ✅ Automated fine-tuning pipeline
- ✅ Web UI for model management
- ✅ API endpoints for inference
- ✅ Model versioning and rollback
- ✅ Cost optimization with spot instances
- ✅ Security best practices

## Directory Structure

```
├── terraform/           # Infrastructure as Code
├── docker/             # Docker configurations
├── app/                # Web application
├── pipeline/           # Fine-tuning pipeline
├── scripts/            # Deployment scripts
├── monitoring/         # Monitoring configurations
└── docs/              # Documentation
```

## Fine-tuning Pipeline

The fine-tuning pipeline supports:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Full fine-tuning
- Custom datasets
- Hyperparameter optimization

## Monitoring

- Model performance metrics
- Resource utilization
- Cost tracking
- Training progress
- API usage analytics

## Security

- VPC with private subnets
- IAM roles with least privilege
- Secrets management with AWS Secrets Manager
- Network ACLs and security groups
- SSL/TLS encryption

## Cost Optimization

- Spot instances for training
- Auto-scaling based on demand
- S3 lifecycle policies
- CloudWatch cost alerts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 