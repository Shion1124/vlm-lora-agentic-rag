# ============================================================
# Dockerfile: VLM + Visual RAG + Agentic RAG API (Cloud Run)
#
# Cloud Run デプロイ用（CPUモード / GPU なし）
# GPU 環境では deployment/Dockerfile を使用してください
#
# Deploy:
#   gcloud run deploy vlm-agentic-rag-api --source . --region us-central1
#
# Local:
#   docker build -t vlm-agentic-rag:latest .
#   docker run -p 8080:8080 vlm-agentic-rag:latest
# ============================================================

FROM python:3.10-slim

# ============================================================
# System Setup
# ============================================================

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libpoppler-cpp-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Dependencies
# ============================================================

COPY requirements_cloudrun.txt .

# CPU版 torch をインストール（Cloud Run は GPU なし）
RUN pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_cloudrun.txt

# ============================================================
# Application Code
# ============================================================

COPY src/api_production.py .
COPY src/vlm_agentic_rag_complete.py .

# ============================================================
# Non-root user
# ============================================================

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# ============================================================
# Health Check & Run
# ============================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api_production:app --host 0.0.0.0 --port ${PORT:-8080}"]
