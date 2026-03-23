---
title: "FastAPI + Docker で本番環境化｜Cloud Run デプロイ"
description: "FastAPI v2.0.0 + Docker による本番環境構築＆Google Cloud Run デプロイ完全ガイド。APIキー認証、CORS制限、Visual RAG + Agentic RAG マルチモーダル対応、実際のデプロイ経験とトラブルシューティングまで全ステップ解説。"
category: "インフラ/デプロイメント"
tags: ["FastAPI", "Docker", "Google Cloud Run", "APIキー認証", "Visual RAG", "Agentic RAG", "本番環境", "API"]
date: "2026-03-21"
author: "Yoshihisa Shinzaki"
slug: "fastapi-docker-cloud-run-deployment"
---

# FastAPI + Docker で本番環境化｜Cloud Run デプロイ

## はじめに

VLM + LoRA + Visual RAG + Agentic RAG を実装したモデルをローカルで動かしても、実用的ではありません。

本記事では、**FastAPI v2.0.0 + Docker + Google Cloud Run** を用いて、**APIキー認証・マルチモーダル対応の本番環境** を構築し、実際にデプロイするまでの全工程を解説します。

```
【このアプローチの利点】
✅ 自動スケーリング（トラフィック対応）
✅ サーバーレス（運用コスト最小化）
✅ APIキー認証（X-API-Key ヘッダー） 🆕
✅ Visual RAG + Agentic RAG マルチモーダル対応 🆕
✅ REST API で容易にアクセス
✅ `gcloud run deploy --source .` でシンプルデプロイ
```

---

## 目次

- [FastAPI とは](#FastAPI-とは)
- [ローカル環境での実装](#ローカル環境での実装)
- [APIキー認証の実装](#APIキー認証の実装)
- [Docker コンテナ化](#Docker-コンテナ化)
- [Google Cloud Run へデプロイ](#Google-Cloud-Run-へデプロイ)
- [実際のデプロイ結果](#実際のデプロイ結果)
- [運用とモニタリング](#運用とモニタリング)
- [トラブルシューティング](#トラブルシューティング)

---

## FastAPI とは

### FastAPI の特徴

```python
"""
FastAPI = 高速 + シンプル + 本番対応の Web フレームワーク

┌─────────────────────────────────────────────┐
│ FastAPI: Uvicorn + Starlette + Pydantic   │
├─────────────────────────────────────────────┤
│ ✅ 自動的に OpenAPI ドキュメント生成        │
│ ✅ 入力検証 (Pydantic)                     │
│ ✅ 非同期処理対応 (async/await)             │
│ ✅ デフォルトで JSON レスポンス対応         │
│ ✅ TypeScript/OpenAPI スキーマ対応          │
│ ✅ 毎秒数千リクエでも対応可能               │
└─────────────────────────────────────────────┘
"""
```

### Flask との比較

```
┌──────────────────┬─────────────┬──────────────┐
│ 項目              │ Flask       │ FastAPI      │
├──────────────────┼─────────────┼──────────────┤
│ 自動ドキュメント  │ ❌ 手動作成 │ ✅ 自動生成   │
│ パフォーマンス    │ 中～低       │ ✅ 高速       │
│ 非同期サポート    │ ⚠️ 限定     │ ✅ 完全対応   │
│ 入力検証         │ ❌ 手動      │ ✅ 自動      │
│ 本番対応         │ ⚠️ 要設定    │ ✅ すぐ対応  │
└──────────────────┴─────────────┴──────────────┘
```

---

## ローカル環境での実装

### ステップ 1: FastAPI アプリケーション構築

```python
# src/api_production.py (v2.0.0)

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import logging
import os

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# API キー認証
# ============================================================
API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """API キーを検証（環境変数 API_KEY が設定されている場合のみ）"""
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# Pydantic モデル（入力検証）
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class MultimodalSearchRequest(BaseModel):
    query: str
    top_k: int = 5

# FastAPI アプリケーション初期化
app = FastAPI(
    title="VLM + LoRA + Visual RAG + Agentic RAG API",
    description="Multimodal document structuring: Visual RAG (image search) + Agentic RAG (text search)",
    version="2.0.0"
)

# CORS ミドルウェア設定（本番環境ではオリジンを制限）
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://*.run.app,http://localhost:3000,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# エンドポイント（GET は認証不要、POST は認証必須）

@app.get("/")
async def root():
    """API 情報（認証不要）"""
    return {
        "name": "VLM + LoRA + Visual RAG + Agentic RAG API",
        "version": "2.0.0",
        "architecture": "Multimodal (Visual + Text) RAG",
        "endpoints": {
            "/docs": "Swagger UI documentation",
            "/health": "Health check",
            "/analyze": "POST Document analysis (認証必須)",
            "/search": "POST Text search (認証必須)",
            "/multimodal-search": "POST Multimodal search (認証必須)"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック（認証不要）"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "visual_rag_available": hasattr(pipeline, 'visual_rag') and pipeline.visual_rag is not None
    }

@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search_documents(request: SearchRequest):
    """Agentic RAG を用いたテキスト検索（認証必須）"""
    results = pipeline.rag.search(request.query, top_k=request.top_k)
    return results

@app.post("/multimodal-search", dependencies=[Depends(verify_api_key)])
async def multimodal_search(request: MultimodalSearchRequest):
    """Visual RAG + Agentic RAG マルチモーダル検索（認証必須）"""
    results = pipeline.multimodal_search(request.query, top_k=request.top_k)
    return results

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_document(file: UploadFile = File(...)):
    """PDFまたは画像ファイルを分析して構造化JSON出力（認証必須）"""
    # ... 実装
    pass
```

### APIキー認証の仕組み

```
【認証フロー】
クライアント → X-API-Key ヘッダー付き → FastAPI
                                          │
                                    verify_api_key()
                                          │
                              ┌───────────┴───────────┐
                           有効                    無効/なし
                              │                       │
                         処理続行                403 Forbidden

【設定方法】
# Cloud Run の環境変数で設定
gcloud run services update vlm-agentic-rag-api \
  --region us-central1 \
  --set-env-vars "API_KEY=your-secure-random-key"

# API_KEY が空の場合は認証スキップ（開発モード）
```

### CORS制限

```python
# 本番環境: 特定オリジンのみ許可
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://*.run.app,http://localhost:3000"
).split(",")

# ❌ 本番環境では allow_origins=["*"] は使わない
# ✅ 環境変数で制御
```

### ステップ 2: ローカルでの実行テスト

```bash
# 依存パッケージをインストール
pip install fastapi uvicorn pydantic

# ローカルサーバーを起動
python src/api_production.py

# 別のターミナルでテスト
curl http://localhost:8000/health

# ブラウザで OpenAPI ドキュメント確認
# http://localhost:8000/docs にアクセス
```

**出力例**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## Docker コンテナ化

### ステップ 1: Dockerfile の作成

Cloud Run（CPU）と GPU環境でDockerfileを分けています。

#### Cloud Run 用 Dockerfile（ルート）

```dockerfile
# Dockerfile (Cloud Run / CPU)
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# システムパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libpoppler-cpp-dev poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU版 torch を先にインストール（Cloud Run は GPU なし）
COPY requirements_cloudrun.txt .
RUN pip install --no-cache-dir torch==2.2.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_cloudrun.txt

# アプリケーションコード
COPY src/api_production.py .
COPY src/vlm_agentic_rag_complete.py .

# セキュリティ: 非root ユーザーで実行
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

EXPOSE 8080
CMD ["sh", "-c", "uvicorn api_production:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

**ポイント:**
- `python:3.10-slim`（Cloud Run は CPU のみ、`nvidia/cuda` は不可）
- `torch==2.2.2+cpu` で CPU 版を明示指定
- `libgl1`（`libgl1-mesa-glx` は Debian Trixie で廃止）
- 非root ユーザー `appuser` でセキュリティ強化

#### GPU 環境用 Dockerfile

```dockerfile
# deployment/Dockerfile (GPU 環境)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 環境変数設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# パッケージ更新＆インストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存パッケージをインストール
COPY deployment/requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# ソースコードをコピー
COPY src/ ./src
COPY files/ ./files

# ポート公開（Cloud Run は PORT 環境変数を使用）
EXPOSE 8080

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8080}/health')" || exit 1

# アプリケーション起動
# Cloud Run では PORT 環境変数を使用
CMD exec uvicorn src.api_production:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 4
```

### ステップ 2: requirements.txt の準備

```
# deployment/requirements_production.txt

torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
transformers==4.38.0
peft==0.8.0
bitsandbytes==0.42.0
huggingface-hub>=0.16.0
accelerate==0.27.0
fastapi==0.104.1
uvicorn==0.24.0
pillow>=9.0.0
numpy<2
pydantic>=2.0
python-multipart>=0.0.6
```

### ステップ 3: ローカルで Docker イメージをビルド

```bash
# イメージをビルド
docker build -t vlm-agentic-rag:latest -f deployment/Dockerfile .

# ローカルで実行テスト
docker run -p 8000:8000 vlm-agentic-rag:latest

# 別のターミナルでテスト
curl http://localhost:8000/health
```

**出力例**:
```
Step 1/10 : FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
...
Successfully built abc123def456
Successfully tagged vlm-agentic-rag:latest
```

---

## Google Cloud Run へデプロイ

### 方法 1: `--source` デプロイ（推奨・最もシンプル）

ローカルの Docker Desktop 不要。Google Cloud Build がクラウド上でビルドします。

```bash
# GCP プロジェクト設定
gcloud config set project vlm-agentic-rag

# ワンコマンドでデプロイ（ルートの Dockerfile を自動検出）
gcloud run deploy vlm-agentic-rag-api \
    --source . \
    --region us-central1 \
    --memory 16Gi \
    --cpu 4 \
    --timeout 3600 \
    --set-env-vars "API_KEY=your-secure-key" \
    --allow-unauthenticated
```

**フラグの説明**:
```
--source .              # カレントディレクトリの Dockerfile でビルド
--memory=16Gi           # メモリ: 16GB（モデル用）
--cpu=4                 # CPU: 4 コア
--timeout=3600          # タイムアウト: 60 分（推論用）
--set-env-vars          # 環境変数（API_KEY 設定）
--allow-unauthenticated # 公開アクセス許可（API キーで制御）
```

### 方法 2: Artifact Registry 経由（従来方式）

```bash
# プロジェクト設定
export PROJECT_ID="vlm-agentic-rag"
export SERVICE_NAME="vlm-agentic-rag-api"
export REGION="us-central1"

# Artifact Registry 有効化
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com

# リポジトリ作成
gcloud artifacts repositories create docker-repo \
    --repository-format=docker \
    --location=$REGION
```

### ステップ 2: Docker イメージを Artifact Registry にプッシュ

```bash
# 認証設定
gcloud auth configure-docker $REGION-docker.pkg.dev

# イメージをタグ付け
docker tag vlm-agentic-rag:latest \
    $REGION-docker.pkg.dev/$PROJECT_ID/docker-repo/vlm-agentic-rag:latest

# プッシュ
docker push $REGION-docker.pkg.dev/$PROJECT_ID/docker-repo/vlm-agentic-rag:latest
```

### ステップ 3: Cloud Run にデプロイ

```bash
# Cloud Run にデプロイ
gcloud run deploy $SERVICE_NAME \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/docker-repo/vlm-agentic-rag:latest \
    --region=$REGION \
    --platform=managed \
    --memory=16Gi \
    --cpu=4 \
    --timeout=3600 \
    --max-instances=10 \
    --min-instances=1 \
    --allow-unauthenticated
```

**フラグの説明**:
```
--memory=16Gi           # メモリ: 16GB（モデル用）
--cpu=4                 # CPU: 4 コア
--timeout=3600          # タイムアウト: 60 分（推論用）
--max-instances=10      # 最大インスタンス数（自動スケーリング）
--min-instances=1       # 最小インスタンス数（常時稼働）
--allow-unauthenticated # 認証なしで公開アクセス許可
```

**出力例**:
```
Service [vlm-agentic-rag-api] revision [vlm-agentic-rag-api-001] has been deployed and is serving 100 percent of traffic.

Service URL: https://vlm-agentic-rag-api-744279114226.us-central1.run.app
```

### デプロイ確認

```bash
# ヘルスチェック
curl https://vlm-agentic-rag-api-744279114226.us-central1.run.app/health

# API ドキュメント（Swagger UI）
# ブラウザで以下にアクセス
# https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs

# APIキー付きテスト
curl -X POST https://vlm-agentic-rag-api-744279114226.us-central1.run.app/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-key" \
  -d '{"query": "VLMの仕組み"}'

# APIキーなしだと拒否される
curl -X POST https://vlm-agentic-rag-api-744279114226.us-central1.run.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
# → {"detail": "Invalid or missing API key"}
```

---

## 実際のデプロイ結果

### デプロイ成功時の出力

```
Service [vlm-agentic-rag-api] revision [vlm-agentic-rag-api-00004-rb7]
has been deployed and is serving 100 percent of traffic.

Service URL: https://vlm-agentic-rag-api-744279114226.us-central1.run.app
```

### ヘルスチェック結果

```json
{
    "status": "healthy",
    "model_loaded": false,
    "base_model": "llava-v1.5-7b",
    "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
    "lora_loaded": false,
    "visual_rag_available": false,
    "timestamp": "2026-03-23T08:30:46.278192"
}
```

**補足:**
- `model_loaded: false` → Cloud Run は CPU 環境のため、LLaVA はモックモードで動作（想定通り）
- `visual_rag_available: false` → `HF_TOKEN` 環境変数を設定すれば CLIP が有効化される
- GPU 環境（Colab/ローカルGPU）では全てのモデルが完全に動作

### ルートエンドポイント結果

```json
{
    "name": "VLM + LoRA + Visual RAG + Agentic RAG API",
    "version": "2.0.0",
    "architecture": "Multimodal (Visual + Text) RAG",
    "endpoints": {
        "/docs": "Swagger UI documentation",
        "/health": "Health check (includes Visual RAG status)",
        "/analyze": "POST Document analysis (with LoRA fine-tuning)",
        "/search": "POST Text search (Agentic RAG only)",
        "/multimodal-search": "POST Multimodal search (Visual RAG + Agentic RAG)"
    }
}
```

### 環境変数の管理

```bash
# 環境変数を確認
gcloud run services describe vlm-agentic-rag-api \
  --region=us-central1 \
  --format='table(spec.template.spec.containers[0].env[].name, spec.template.spec.containers[0].env[].value)'

# APIキーを安全な値に更新
gcloud run services update vlm-agentic-rag-api \
  --region us-central1 \
  --set-env-vars "API_KEY=$(openssl rand -base64 32)"

# HuggingFace トークンも追加（Visual RAG 有効化用）
gcloud run services update vlm-agentic-rag-api \
  --region us-central1 \
  --set-env-vars "API_KEY=your-key,HF_TOKEN=hf_xxxxxxxxxxxxx"
```

---

## 運用とモニタリング

### ステップ 1: ログ確認

```bash
# Cloud Run のログを確認
gcloud run logs read $SERVICE_NAME --region=$REGION

# リアルタイムログ
gcloud run logs read $SERVICE_NAME --region=$REGION --stream

# 特定エラーを抽出
gcloud run logs read $SERVICE_NAME --region=$REGION | grep ERROR
```

### ステップ 2: パフォーマンスモニタリング

```python
# src/monitoring.py

import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Request
from fastapi.responses import Response

# メトリクス定義
request_count = Counter(
    'vlm_requests_total',
    'Total number of requests',
    ['method', 'endpoint']
)

request_duration = Histogram(
    'vlm_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

model_load_time = Histogram(
    'vlm_model_load_seconds',
    'Model load time in seconds'
)

# ミドルウェア
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    request_duration.labels(
        endpoint=request.url.path
    ).observe(process_time)
    
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus メトリクス公開"""
    return Response(content=generate_latest(), media_type="text/plain")
```

### ステップ 3: アラート設定

```bash
# Cloud Monitoring でアラートを作成
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="VLM API - High Latency Alert" \
    --condition-display-name="P95 latency > 5s"
```

---

## トラブルシューティング

### 問題 1: Buildpacks によるビルド失敗

```
ERROR: failed to build: ... unable to determine application file type
```

**原因**: プロジェクトルートに Dockerfile がない場合、Cloud Build は Buildpacks を使用するが、torch 等の ML 依存関係に対応できない。

**解決策**:
```bash
# ルートに Dockerfile を配置
# Cloud Run は --source . で Dockerfile を自動検出
ls Dockerfile  # ← これが必要
gcloud run deploy --source .
```

### 問題 2: libgl1-mesa-glx が見つからない

```
E: Package 'libgl1-mesa-glx' has no installation candidate
```

**原因**: `python:3.10-slim` の最新版は Debian Trixie ベースで、`libgl1-mesa-glx` が廃止された。

**解決策**:
```dockerfile
# ❌ Debian Trixie で廃止
RUN apt-get install -y libgl1-mesa-glx

# ✅ 代替パッケージ
RUN apt-get install -y libgl1
```

### 問題 3: nvidia/cuda イメージが Cloud Run で動かない

```
Cloud Run は CPU 環境のため、CUDA ベースイメージは不適切
```

**解決策**:
```dockerfile
# ❌ Cloud Run では使えない
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ✅ Cloud Run 用
FROM python:3.10-slim

# CPU版 torch を明示的にインストール
RUN pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# bitsandbytes は CUDA 必須のため除外
# requirements_cloudrun.txt から bitsandbytes を削除
```

### 問題 4: ポートバインディングエラー

```
OSError: [Errno 48] Address already in use
```

**解決策**:
```bash
# Dockerfile で PORT 環境変数を使用
CMD exec uvicorn src.api_production:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080}

# Cloud Run は自動的に PORT=8080 を設定
```

### 問題 2: メモリ不足エラー

```
RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB
```

**解決策**:
```bash
# メモリを増加
gcloud run deploy $SERVICE_NAME \
    --image=... \
    --memory=20Gi  # 20GB に増加

# メモリ削減オプション
# 4-bit 量子化を確認
# --max-instances を削減
```

### 問題 3: デプロイ時間が長い

```
Building and pushing image (this may take up to 20 minutes)...
```

**解決策**:
```bash
# ローカルでビルドして直接プッシュ
docker build -t vlm-agentic-rag .
docker tag vlm-agentic-rag $REGION-docker.pkg.dev/$PROJECT_ID/docker-repo/...
docker push $REGION-docker.pkg.dev/$PROJECT_ID/docker-repo/...

# より高速なデプロイ
gcloud run deploy --image=... --source=.
```

### 問題 4: モデルロードが遅い（Cold Start）

```
推論までに 30-60 秒かかる
```

**解決策**:
```bash
# min-instances を増加（常時稼働インスタンスを確保）
gcloud run deploy $SERVICE_NAME \
    --image=... \
    --min-instances=2  # 最小 2 インスタンス常時稼働

# モデルをキャッシュ
# Dockerfile に以下を追加
RUN python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('llava-hf/llava-1.5-7b-hf')"
```

---

## CI/CD オートメーション

### GitHub Actions による自動デプロイ

```yaml
# .github/workflows/deploy.yml

name: Deploy to Cloud Run

on:
  push:
    branches:
      - main
    paths:
      - 'src/**'
      - 'deployment/**'
      - '.github/workflows/**'

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  SERVICE_NAME: vlm-agentic-rag-api

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Google Cloud
        uses: google-github-actions/setup-gcloud@v1
        with:
          project_id: ${{ env.PROJECT_ID }}
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          export_default_credentials: true
      
      - name: Configure Docker for Artifact Registry
        run: |
          gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev
      
      - name: Build and push image
        run: |
          docker build -t ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/docker-repo/vlm-agentic-rag:latest .
          docker push ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/docker-repo/vlm-agentic-rag:latest
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image=${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/docker-repo/vlm-agentic-rag:latest \
            --region=${{ env.REGION }} \
            --memory=16Gi \
            --cpu=4 \
            --allow-unauthenticated
```

---

## 参考文献

### FastAPI と Web Application

1. **FastAPI: The Modern, Fast Web Framework**  
   Ramírez, S. (2018-Present)  
   https://fastapi.tiangolo.com/  
   FastAPI の公式ドキュメント

2. **Asynchronous Python and FastAPI**  
   https://fastapi.tiangolo.com/async-tests/  
   非同期処理のベストプラクティス

### Docker と コンテナ化

3. **Docker Best Practices**  
   Docker Inc.  
   https://docs.docker.com/develop/dev-best-practices/  
   Docker のベストプラクティス

4. **Dockerfile Reference**  
   https://docs.docker.com/engine/reference/builder/  
   Dockerfile の詳細なリファレンス

### Google Cloud Run

5. **Google Cloud Run Documentation**  
   Google Cloud  
   https://cloud.google.com/run/docs  
   Cloud Run の公式ドキュメント

6. **Cloud Run Best Practices**  
   https://cloud.google.com/run/docs/quickstarts/build-and-deploy  
   本番環境でのベストプラクティス

### CI/CD と自動デプロイ

7. **GitHub Actions for Google Cloud**  
   https://github.com/google-github-actions/setup-gcloud  
   GitHub Actions によるスムーズなデプロイ

---

## チェックリスト

```
デプロイメント前：
☐ 1. FastAPI アプリケーションをローカルで動作確認
☐ 2. Dockerfile をビルドして実行確認
☐ 3. GCP プロジェクトを初期化
☐ 4. Artifact Registry リポジトリを作成

デプロイメント実行：
☐ 5. Docker イメージをビルド
☐ 6. イメージを Artifact Registry にプッシュ
☐ 7. Cloud Run にデプロイ
☐ 8. ヘルスチェックで動作確認
☐ 9. API ドキュメントにアクセス確認

運用：
☐ 10. ログをモニタリング
☐ 11. パフォーマンスメトリクスを確認
☐ 12. GitHub Actions CI/CD を設定
☐ 13. Alert ルールを定義
```

---

## まとめ

FastAPI v2.0.0 + Docker + Cloud Run で、**APIキー認証付きの自動スケールする本番環境** を実現できます。

```
【このアプローチの価値】
✅ サーバーレスで運用コスト最小化
✅ APIキー認証でセキュリティ確保
✅ CORS制限でオリジン管理
✅ Visual RAG + Agentic RAG のマルチモーダルAPI
✅ `gcloud run deploy --source .` でシンプルデプロイ
✅ 非rootユーザーでコンテナ実行
```

次の記事では、「**パフォーマンス最適化｜4-bit 量子化で 50% メモリ削減**」で、本番環境での最適化テクニックを解説します。

では、次の記事でお会いしましょう！

---

## 関連リンク

- 📘 [GitHub リポジトリ](https://github.com/Shion1124/vlm-lora-agentic-rag)
- 🤗 [HuggingFace モデル](https://huggingface.co/Shion1124/vlm-lora-agentic-rag)
- 🚀 [ライブ API](https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs)
- 📚 [前の記事: Agentic RAG とは何か](#)
- 📚 [次の記事: パフォーマンス最適化](#)
- 🔧 [FastAPI Documentation](https://fastapi.tiangolo.com/)
- 🐳 [Docker Documentation](https://docs.docker.com/)
- ☁️ [Google Cloud Run](https://cloud.google.com/run)

---

**更新履歴**

- 2026-03-21：初版公開
- 2026-03-23：API v2.0.0に更新（APIキー認証・CORS制限・Visual RAG対応）、実際のデプロイ結果・トラブルシューティング追加、Cloud Run CPU版Dockerfile追加

---

**著者情報**

Yoshihisa Shinzaki  
Machine Learning Engineer | Infrastructure & Deployment Specialist

関心領域：本番環境デプロイメント、FastAPI、コンテナ化、自動スケーリング、APIセキュリティ
