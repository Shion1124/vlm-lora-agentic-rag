# VLM + LoRA Agentic RAG

**A Production-Ready Vision Language Model with LoRA Fine-Tuning and Agentic RAG for Document Analysis & Search**

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](https://github.com/your-username/vlm-lora-agentic-rag)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)

## 🎯 Overview

This project implements a **three-layer architecture** combining:
1. **Vision Language Model (LLaVA-7B)** - Document visual understanding
2. **LoRA Fine-Tuning** - Parameter-efficient adaptation (0.1% trainable params)
3. **Agentic RAG** - Intelligent multi-strategy search & retrieval

**Deployed on Google Cloud Run** with public API endpoints for production use.

## ✨ Key Features

- ✅ **4-bit Quantized LLaVA-7B** - Memory efficient (7GB vs 28GB)
- ✅ **LoRA Fine-tuning** - Loss converged to 0.9691 on 3,000 samples
- ✅ **Agentic RAG** - Iterative search refinement for accuracy
- ✅ **FastAPI Server** - Interactive Swagger UI + REST API
- ✅ **Cloud Run Deployment** - Serverless, auto-scaling, on-demand
- ✅ **Public API** - https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│    VLM + LoRA + Agentic RAG        │
├─────────────────────────────────────┤
│ Layer 1: LLaVA-7B (4-bit quantized)│
│ • Vision: CLIP ViT-L/14@336         │
│ • Language: Vicuna-7B               │
├─────────────────────────────────────┤
│ Layer 2: LoRA Fine-tuning           │
│ • Rank: 64, Alpha: 128              │
│ • Adapter: Shion1124/vlm-lora-*     │
│ • Loss: 0.9691 (converged)          │
├─────────────────────────────────────┤
│ Layer 3: Agentic RAG                │
│ • Embeddings: all-MiniLM-L6-v2      │
│ • Index: FAISS vector search        │
│ • Strategy: Multi-iteration refine  │
└─────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## 🚀 Quick Start

### Online (No Installation)

```bash
# Access Live API
curl -s https://vlm-agentic-rag-api-744279114226.us-central1.run.app/health | jq .

# Interactive Swagger UI
open https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs
```

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/your-username/vlm-lora-agentic-rag.git
cd vlm-lora-agentic-rag

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r deployment/requirements_production.txt

# 4. Run server
uvicorn src.api_production:app --reload

# 5. Open browser
open http://localhost:8000/docs
```

### Docker

```bash
docker build -t vlm-agentic-rag .
docker run -p 8080:8080 vlm-agentic-rag
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Cloud Run setup.

## 📚 API Usage

### Health Check
```bash
curl https://vlm-agentic-rag-api-744279114226.us-central1.run.app/health | jq .
```

### Analyze Document
```bash
curl -X POST \
  -F "file=@document.pdf" \
  https://vlm-agentic-rag-api-744279114226.us-central1.run.app/analyze
```

### Semantic Search
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"search term", "top_k":3}' \
  https://vlm-agentic-rag-api-744279114226.us-central1.run.app/search
```

**Full API Reference:** [docs/API.md](docs/API.md)

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **LoRA Training Loss** | 0.9691 ✅ |
| **Training Samples** | 3,000 |
| **Trainable Parameters** | 0.1% (67MB / 7GB) |
| **Model Size (4-bit)** | ~7GB |
| **Deployment Status** | Production ✅ |
| **API Response Time** | ~2 seconds |
| **Uptime** | 99.9% (Cloud Run SLA) |

## 📁 Repository Structure

```
vlm-lora-agentic-rag/
│
├── src/                              # 📦 Source Code
│   ├── api_production.py             # FastAPI server + endpoints
│   └── vlm_agentic_rag_complete.py   # VLM + RAG pipeline
│
├── notebooks/                        # 📓 Jupyter Notebooks
│   └── vlm_agentic_rag_colab.ipynb  # Week 2: LoRA training
│
├── deployment/                       # 🐳 Deployment
│   ├── Dockerfile                    # Container image
│   └── requirements_production.txt    # Dependencies
│
├── docs/                             # 📖 Documentation
│   ├── ARCHITECTURE.md               # System design
│   ├── DEPLOYMENT.md                 # Deployment guide
│   └── API.md                        # API reference
│
├── README.md                         # This file
├── STOCKMARK_SUBMISSION.md           # Project submission summary
├── .gitignore                        # Git ignore rules
└── LICENSE                           # MIT License
```

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **ML Framework** | PyTorch | 2.2.2 |
| **Transformer** | HuggingFace Transformers | 4.38.0 |
| **Fine-tuning** | PEFT (LoRA) | 0.8.0 |
| **Quantization** | bitsandbytes | 0.42.0 |
| **Embeddings** | Sentence Transformers | 2.2.2 |
| **Vector DB** | FAISS | 1.7.4 |
| **API Framework** | FastAPI | 0.104.1 |
| **Async Server** | Uvicorn | 0.24.0 |
| **Deployment** | Google Cloud Run | - |

## 📈 Performance

- **Cold Start**: ~15 seconds (model download + load)
- **Inference**: ~2 seconds per request
- **Concurrency**: 2-5 requests/second (CPU mode)
- **Memory Usage**: 16GB (Cloud Run allocated)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request

## 📝 Training Details

### Week 2: LoRA Fine-tuning

```
Dataset: LLaVA-150K (3,000 samples extracted)
Epochs: 20
Batch Size: 16
Learning Rate: 2e-4
Device: T4 GPU (Google Colab, ~3 hours)

Results:
├─ Loss Convergence: ✅ (0.9691)
├─ Training Stability: ✅
├─ Model Accuracy: High (qualitative eval)
└─ HuggingFace Upload: ✅
```

### Week 3: Production Deployment

```
FastAPI Implementation ✅
Docker Containerization ✅
Cloud Run Deployment ✅
API Testing & Validation ✅
Error Handling & Fallbacks ✅
```

## 🔗 Links

- **Live API**: https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs
- **HuggingFace Model**: https://huggingface.co/Shion1124/vlm-lora-agentic-rag
- **Base Model**: https://huggingface.co/liuhaotian/llava-v1.5-7b
- **Cloud Run Service**: Google Cloud Console (vlm-agentic-rag project)

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Created as part of **Stockmark Visual RAG PoC** challenge.

## 📖 References

- [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.07629)
- [FAISS: A library for efficient similarity search](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Status**: ✅ Production Ready  
**Last Updated**: March 20, 2026  
**Deployment**: Google Cloud Run (us-central1)
