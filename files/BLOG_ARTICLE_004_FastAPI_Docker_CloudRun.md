---
title: "FastAPI + Docker で本番環境化｜Cloud Run デプロイ"
description: "FastAPI + Docker による本番環境構築＆Google Cloud Run デプロイ完全ガイド。環境構築から自動スケール設定、トラブルシューティングまで全ステップ解説。"
category: "インフラ/デプロイメント"
tags: ["FastAPI", "Docker", "Google Cloud Run", "CI/CD", "本番環境", "API"]
date: "2026-03-21"
author: "Yoshihisa Shinzaki"
slug: "fastapi-docker-cloud-run-deployment"
---

# FastAPI + Docker で本番環境化｜Cloud Run デプロイ

## はじめに

VLM + LoRA + Agentic RAG を実装したモデルをローカルで動かしても、実用的ではありません。

本記事では、**FastAPI + Docker + Google Cloud Run** を用いて、**完全にサーバーレスな本番環境** を構築する方法を解説します。

```
【このアプローチの利点】
✅ 自動スケーリング（トラフィック対応）
✅ サーバーレス（運用コスト最小化）
✅ REST API で容易にアクセス
✅ GitHub Actions による自動デプロイ
```

---

## 目次

- [FastAPI とは](#FastAPI-とは)
- [ローカル環境での実装](#ローカル環境での実装)
- [Docker コンテナ化](#Docker-コンテナ化)
- [Google Cloud Run へデプロイ](#Google-Cloud-Run-へデプロイ)
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
# src/api_production.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
import torch
from typing import Optional
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic モデル（入力検証）
class AnalysisRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 200

class AnalysisResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    processing_time: float

# FastAPI アプリケーション初期化
app = FastAPI(
    title="VLM Agentic RAG API",
    description="Vision Language Model + Agentic RAG による画像＆テキスト分析",
    version="1.0.0"
)

# CORS ミドルウェア設定（全オリジンを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバルモデル（初回起動時ロード）
model = None
processor = None
device = None

@app.on_event("startup")
async def load_model():
    """アプリケーション起動時にモデルをロード"""
    global model, processor, device
    
    logger.info("Loading model...")
    
    try:
        # デバイス設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # ベースモデル
        base_model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA adapter をロード
        lora_model_id = "Shion1124/vlm-lora-agentic-rag"
        model = PeftModel.from_pretrained(model, lora_model_id)
        
        # Processor
        processor = AutoProcessor.from_pretrained(base_model_id)
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # フォールバック: モデルなしで起動
        model = None
        processor = None

@app.on_event("shutdown")
async def cleanup():
    """アプリケーション終了時にメモリをクリア"""
    global model, processor
    del model
    del processor
    torch.cuda.empty_cache()
    logger.info("Model unloaded")

# エンドポイント

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    request: AnalysisRequest,
    image_file: Optional[UploadFile] = File(None)
):
    """
    画像ファイルとテキストプロンプトから分析結果を生成
    
    - image_file: アップロードされた画像ファイル
    - request: プロンプト、温度パラメータなど
    """
    import time
    
    start_time = time.time()
    
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if image_file is None:
            raise HTTPException(status_code=400, detail="Image file required")
        
        # 画像を読み込み
        from PIL import Image
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 入力処理
        inputs = processor(
            text=request.prompt,
            images=image,
            return_tensors="pt"
        ).to(device)
        
        # 推論
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        
        # 出力をデコード
        response_text = processor.decode(output[0], skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            response=response_text,
            confidence=0.85,  # 簡略版（実装では動的に計算）
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(query: str, top_k: int = 5):
    """
    Agentic RAG を用いたドキュメント検索
    """
    # 実装詳細は省略
    return {
        "query": query,
        "results": [],
        "strategy": "agentic"
    }

@app.get("/docs")
async def get_docs():
    """
    OpenAPI ドキュメント（自動生成）
    http://localhost:8000/docs にアクセス
    """
    pass

if __name__ == "__main__":
    import uvicorn
    
    # ローカルで実行
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
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

```dockerfile
# Dockerfile

# マルチステージビルド（最終イメージサイズを削減）
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

### ステップ 1: Google Cloud プロジェクト初期化

```bash
# GCP プロジェクト設定
export PROJECT_ID="your-project-id"
export SERVICE_NAME="vlm-agentic-rag-api"
export REGION="us-central1"

# GCP にサインイン
gcloud auth login

# プロジェクト設定
gcloud config set project $PROJECT_ID

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

### ステップ 4: デプロイ確認

```bash
# ヘルスチェック
curl https://vlm-agentic-rag-api-744279114226.us-central1.run.app/health

# ドキュメント確認
# ブラウザで以下にアクセス
# https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs
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

### 問題 1: ポートバインディングエラー

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

FastAPI + Docker + Cloud Run で、**自動スケールする本番環境** を実現できます。

```
【このアプローチの価値】
✅ サーバーレスで運用コスト最小化
✅ 自動スケーリングで トラフィック対応
✅ REST API で容易にアクセス
✅ GitHub Actions で自動デプロイ
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

---

**著者情報**

Yoshihisa Shinzaki  
Machine Learning Engineer | Infrastructure & Deployment Specialist

関心領域：本番環境デプロイメント、FastAPI、コンテナ化、自動スケーリング
