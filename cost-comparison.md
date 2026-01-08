# GPU Cost Comparison for Deep Learning Training

## Cloud GPU Pricing (as of 2024)

### AWS EC2 Instances

| Instance Type | GPU | vCPUs | Memory | On-Demand $/hr | Spot $/hr (avg) |
|--------------|-----|-------|---------|----------------|-----------------|
| p3.2xlarge | 1x V100 | 8 | 61 GB | $3.06 | ~$0.92 |
| p3.8xlarge | 4x V100 | 32 | 244 GB | $12.24 | ~$3.67 |
| p4d.24xlarge | 8x A100 | 96 | 1152 GB | $32.77 | ~$9.83 |
| g4dn.xlarge | 1x T4 | 4 | 16 GB | $0.526 | ~$0.16 |
| g5.xlarge | 1x A10G | 4 | 16 GB | $1.006 | ~$0.30 |

### Google Cloud Platform

| Machine Type | GPU | vCPUs | Memory | On-Demand $/hr | Preemptible $/hr |
|--------------|-----|-------|---------|----------------|------------------|
| n1-standard-4 + V100 | 1x V100 | 4 | 15 GB | $2.48 | $0.74 |
| n1-standard-8 + V100 | 2x V100 | 8 | 30 GB | $4.96 | $1.48 |
| a2-highgpu-1g | 1x A100 | 12 | 85 GB | $3.67 | $1.10 |
| n1-standard-4 + T4 | 1x T4 | 4 | 15 GB | $0.95 | $0.29 |

### Microsoft Azure

| VM Size | GPU | vCPUs | Memory | Pay-as-you-go $/hr | Spot $/hr (avg) |
|---------|-----|-------|---------|-------------------|-----------------|
| NC6s_v3 | 1x V100 | 6 | 112 GB | $3.06 | ~$0.92 |
| NC24s_v3 | 4x V100 | 24 | 448 GB | $12.24 | ~$3.67 |
| ND96asr_v4 | 8x A100 | 96 | 900 GB | $27.20 | ~$8.16 |
| NC4as_T4_v3 | 1x T4 | 4 | 28 GB | $0.526 | ~$0.16 |

### Specialized ML Platforms

| Service | GPU Options | Pricing | Notes |
|---------|-------------|---------|-------|
| Google Colab | T4, P100, V100 | Free - $49.99/mo | Limited session time |
| Paperspace | RTX 4000 - A100 | $0.45 - $3.09/hr | Persistent storage extra |
| Lambda Labs | RTX 6000 - A100 | $0.50 - $1.29/hr | Often out of stock |
| Vast.ai | Various | $0.10 - $2.00/hr | Peer-to-peer marketplace |
| RunPod | RTX 3090 - A100 | $0.34 - $1.99/hr | Serverless and dedicated |

## Cost Optimization Strategies

### 1. Spot/Preemptible Instances
- **Savings**: 60-90% off on-demand pricing
- **Best for**: Fault-tolerant training with checkpointing
- **Example**: V100 on AWS Spot ~$0.92/hr vs $3.06/hr on-demand

### 2. Reserved Instances
- **AWS**: 1-year (40% off), 3-year (60% off)
- **GCP**: 1-year (37% off), 3-year (56% off)
- **Azure**: 1-year (41% off), 3-year (62% off)

### 3. GPU Time Optimization
- **Mixed Precision**: 2-3x speedup with minimal accuracy loss
- **Gradient Accumulation**: Use smaller GPUs with larger effective batch
- **Model Pruning**: Reduce model size by 50-90%

### 4. Development vs Production
- **Development**: Use CPU or cheap GPUs (T4)
- **Experimentation**: Use spot instances
- **Production Training**: Use on-demand or reserved

## Example Cost Calculations

### Training ResNet-50 on ImageNet
- **Dataset**: 1.2M images
- **Epochs**: 90
- **Time on V100**: ~8 hours

| Platform | Instance | Cost |
|----------|----------|------|
| AWS On-Demand | p3.2xlarge | $24.48 |
| AWS Spot | p3.2xlarge | ~$7.36 |
| GCP Preemptible | n1 + V100 | ~$5.92 |
| Colab Pro+ | V100 | $49.99/month |

### Training BERT-Base
- **Dataset**: Wikipedia + BookCorpus
- **Steps**: 1M
- **Time on V100**: ~4 days

| Platform | Instance | Cost |
|----------|----------|------|
| AWS On-Demand | p3.8xlarge | $1,175.04 |
| AWS Spot | p3.8xlarge | ~$352.32 |
| GCP Preemptible | n1 + 4xV100 | ~$142.08 |

## Recommendations by Use Case

### Hobbyist/Learning
- **Google Colab Free**: Best starting point
- **Kaggle Kernels**: Free P100 GPUs
- **Paperspace Free Tier**: Limited but useful

### Research/Prototyping
- **GCP Preemptible**: Best price/performance
- **AWS Spot**: More availability
- **University Clusters**: Often free for students

### Production/Business
- **Reserved Instances**: Predictable costs
- **Multi-cloud**: Avoid vendor lock-in
- **Hybrid**: On-premise + cloud burst

### Large-Scale Training
- **AWS p4d instances**: 8x A100 for massive models
- **GCP TPU pods**: Cost-effective for very large batches
- **Custom clusters**: Consider building if >$10k/month spend

## Hidden Costs to Consider

1. **Data Transfer**
   - Egress: $0.08-0.12/GB
   - Between regions: $0.01-0.02/GB

2. **Storage**
   - SSD: $0.08-0.17/GB/month
   - Standard: $0.02-0.05/GB/month

3. **Idle Time**
   - GPUs billed by second/minute
   - Don't forget to stop instances

4. **Support**
   - AWS: $29-15,000/month
   - GCP: $150-12,500/month

## Money-Saving Tips

1. **Use spot/preemptible** for all non-critical workloads
2. **Right-size instances** - monitor GPU utilization
3. **Implement checkpointing** to handle interruptions
4. **Use managed services** (SageMaker, Vertex AI) for auto-scaling
5. **Schedule training** during off-peak hours
6. **Compare across clouds** - prices vary by region
7. **Consider older GPUs** - V100 often sufficient vs A100
8. **Batch small jobs** - minimize instance startup time