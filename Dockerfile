# ============================================================
# Dockerfile: VLM + LoRA Agentic RAG API
# 
# Build:
#   docker build -t vlm-agentic-rag:latest .
# 
# Run (local):
#   docker run --gpus all -p 8000:8000 vlm-agentic-rag:latest
# 
# Run (background):
#   docker run -d --gpus all -p 8000:8000 --name vlm-api vlm-agentic-rag:latest
# ============================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ============================================================
# System Setup
# ============================================================

WORKDIR /app

# 言語・タイムゾーン設定
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# システムパッケージ
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Dependencies
# ============================================================

WORKDIR /app

COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# ============================================================
# Application Code
# ============================================================

COPY api_production.py .
COPY vlm_agentic_rag_complete.py .

# ============================================================
# Health Check
# ============================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# ============================================================
# Expose & Run
# ============================================================

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api_production:app --host 0.0.0.0 --port ${PORT:-8080}"]
