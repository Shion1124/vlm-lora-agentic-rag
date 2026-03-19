# Deployment Guide

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- Google Cloud CLI (gcloud)
- Google Cloud Account with billing enabled

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/vlm-lora-agentic-rag.git
cd vlm-lora-agentic-rag

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r deployment/requirements_production.txt

# 4. Run FastAPI server locally
uvicorn src.api_production:app --reload --port 8000

# 5. Access Swagger UI
# Open browser: http://localhost:8000/docs
```

## Google Cloud Run Deployment

### Step 1: Set Up GCP Project

```bash
# Authenticate
gcloud auth login

# Create project
gcloud projects create vlm-agentic-rag --name="VLM LoRA Agentic RAG"

# Set project
gcloud config set project vlm-agentic-rag

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Step 2: Enable Billing

1. Go to [GCP Console Billing](https://console.cloud.google.com/billing)
2. Create billing account
3. Link to vlm-agentic-rag project

### Step 3: Deploy to Cloud Run

```bash
cd /path/to/vlm-lora-agentic-rag

gcloud run deploy vlm-agentic-rag-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600
```

### Step 4: Get Service URL

```bash
gcloud run services describe vlm-agentic-rag-api \
  --region us-central1 \
  --format='value(status.url)'
```

## Docker Build (Optional - Local)

```bash
# Build image
docker build -t vlm-agentic-rag:latest .

# Run container
docker run -p 8080:8080 \
  -e PORT=8080 \
  vlm-agentic-rag:latest

# Access: http://localhost:8080/docs
```

## Environment Variables

```bash
# Optional: HuggingFace Token (for private models)
export HF_TOKEN="hf_xxxxxxxxxxxx"

# Optional: Model caching
export TRANSFORMERS_CACHE="/path/to/cache"
export HF_HOME="/path/to/hf-home"
```

## Troubleshooting

### Issue: Container fails to start
**Solution**: Check logs with:
```bash
gcloud run logs read vlm-agentic-rag-api --limit 100
```

### Issue: Timeout on first request
**Solution**: First request downloads models (~5-10 mins). Subsequent requests are faster.

### Issue: Out of Memory (OOM)
**Solution**: Increase Cloud Run memory:
```bash
gcloud run deploy vlm-agentic-rag-api \
  --update \
  --memory 32Gi
```

### Issue: Model not loading
**Solution**: Ensure HuggingFace token is set (if using private model):
```bash
gcloud run deploy vlm-agentic-rag-api \
  --update \
  --set-env-vars HF_TOKEN=your_token
```

## Monitoring & Logs

```bash
# View recent logs
gcloud run logs read vlm-agentic-rag-api --limit 50

# Check service status
gcloud run services describe vlm-agentic-rag-api --region us-central1

# View metrics in Cloud Console
# https://console.cloud.google.com/run/detail/us-central1/vlm-agentic-rag-api
```

## Cost Estimation

### Google Cloud Run Pricing
- **Compute**: $0.00002400 per vCPU-second
- **Memory**: $0.0000025 per GB-second
- **Requests**: Free first 2M/month

### Estimated Monthly Cost (assuming 1000 requests/day)
- Compute (4 CPU, 16GB): ~$10-20/month
- Storage: ~$5/month (models cached)
- **Total**: ~$15-25/month

## Next Steps

1. ✅ Deploy to Cloud Run
2. ✅ Test API endpoints
3. ✅ Monitor performance
4. 🔄 (Optional) Enable GPU for faster inference
5. 🔄 (Optional) Set up auto-scaling

## Additional Resources

- [Google Cloud Run Docs](https://cloud.google.com/run/docs)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
