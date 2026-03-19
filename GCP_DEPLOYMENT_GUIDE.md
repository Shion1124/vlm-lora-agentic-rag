# GCP Cloud Run デプロイガイド - VLM + LoRA Agentic RAG API

## 事前準備

### 1. GCP プロジェクト作成
```bash
# GCP コンソール https://console.cloud.google.com/ にアクセス

# または gcloud CLI で：
gcloud projects create vlm-agentic-rag --set-as-default

# プロジェクトID確認
gcloud config get-value project
```

### 2. gcloud CLI インストール & 認証
```bash
# macOS
brew install google-cloud-sdk

# 初期化
gcloud init

# 認証ブラウザが開く → ログイン → 許可
gcloud auth application-default login

# プロジェクト確認
gcloud config list
```

### 3. 必須 API 有効化
```bash
# Cloud Run API
gcloud services enable run.googleapis.com

# Artifact Registry API
gcloud services enable artifactregistry.googleapis.com

# Container Registry API（オプション、推奨）
gcloud services enable containerregistry.googleapis.com
```

---

## Docker イメージビルド＆プッシュ

### 4. ローカルでビルド（オプション・検証用）
```bash
cd /Users/yoshihisashinzaki/VLM

# イメージビルド
docker build -t vlm-agentic-rag:latest .

# テスト実行（GPU不要の場合）
docker run -p 8000:8000 vlm-agentic-rag:latest

# curl でテスト
curl http://localhost:8000/health
```

### 5. Artifact Registry にプッシュ

#### 5a. リージョン & リポジトリ作成
```bash
# リージョン（例：us-central1）
GCP_REGION="us-central1"
GCP_PROJECT=$(gcloud config get-value project)

# Artifact Registry リポジトリ作成
gcloud artifacts repositories create vlm-repo \
  --repository-format=docker \
  --location=$GCP_REGION \
  --project=$GCP_PROJECT

echo "✅ Repository created at: $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/vlm-repo"
```

#### 5b. Docker 認証設定
```bash
# gcloud で Docker 認証
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
```

#### 5c. イメージタグ & プッシュ
```bash
# イメージタグ付け
IMAGE_URI="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/vlm-repo/vlm-agentic-rag:latest"

docker tag vlm-agentic-rag:latest $IMAGE_URI

# Artifact Registry にプッシュ
docker push $IMAGE_URI

echo "✅ Image pushed to: $IMAGE_URI"
```

---

## Cloud Run デプロイ

### 6. Cloud Run サービス作成
```bash
GCP_REGION="us-central1"
GCP_PROJECT=$(gcloud config get-value project)
IMAGE_URI="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/vlm-repo/vlm-agentic-rag:latest"

# サービスデプロイ
gcloud run deploy vlm-agentic-rag-api \
  --image=$IMAGE_URI \
  --region=$GCP_REGION \
  --platform=managed \
  --allow-unauthenticated \
  --memory=16Gi \
  --cpu=4 \
  --timeout=3600 \
  --max-instances=2 \
  --set-env-vars="HF_HOME=/models,TRANSFORMERS_CACHE=/models" \
  --project=$GCP_PROJECT

# デプロイ完了後の出力例：
# Service URL: https://vlm-agentic-rag-api-abc123.run.app
```

### デプロイパラメータ説明

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| `--memory` | 16Gi | LLaVA 7B (8GB) + LoRA (0.1%) + RAG コンテキスト |
| `--cpu` | 4 | Google Cloud Run の標準設定 |
| `--timeout` | 3600 | PDFは大きいため1時間余裕を設定 |
| `--max-instances` | 2 | コスト抑制＆サブスクリもの試行段階|
| `--allow-unauthenticated` | - | 開発段階・デモ用（本番では削除） |

---

## 展開後の検証

### 7. API エンドポイント確認
```bash
# Cloud Run サービス一覧
gcloud run services list --region=$GCP_REGION

# サービスURL取得
SERVICE_URL=$(gcloud run services describe vlm-agentic-rag-api \
  --region=$GCP_REGION \
  --format='value(status.url)')

echo "🌐 API URL: $SERVICE_URL"
```

### 8. ヘルスチェック
```bash
SERVICE_URL="https://vlm-agentic-rag-api-abc123.run.app"

# ① Health Check
curl -s $SERVICE_URL/health | jq .

# 期待値：
# {
#   "status": "healthy",
#   "model_loaded": true|false,
#   "model_name": "llava-v1.5-7b",
#   "timestamp": "2026-03-20T..."
# }
```

### 9. API テスト
```bash
SERVICE_URL="https://vlm-agentic-rag-api-abc123.run.app"

# ① ドキュメント分析テスト
curl -X POST -F "file=@files/2026_3q_summary_jp.pdf" \
  $SERVICE_URL/analyze \
  | jq .

# ② ドキュメント検索テスト
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "売上", "top_k": 3}' \
  $SERVICE_URL/search \
  | jq .

# ③ API Docs（Swagger UI）
echo "📚 Swagger UI: $SERVICE_URL/docs"
```

---

## トラブルシューティング

### 問題1：Image エラー「invalid reference format」
```bash
# ❌ 原因：IMAGE_URI が間違っている
# ✅ 解決：
GCP_REGION="us-central1"
GCP_PROJECT=$(gcloud config get-value project)
echo "Expected URI: $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/vlm-repo/vlm-agentic-rag:latest"
echo "=== Verify this matches docker push output ==="
```

### 問題2：「Container failed to start」エラー
```bash
# ログ確認
gcloud run logs read vlm-agentic-rag-api --region=$GCP_REGION --limit=50

# 一般的な原因：
# - メモリ不足（16Gi に増やす）
# - LLaVA リポジトリのダウンロード失敗（Dockerfile 確認）
# - requirements の依存エラー（ローカルテスト）
```

### 問題3：「403 Permission denied」
```bash
# Identity and Access Management (IAM) 確認
gcloud projects get-iam-policy $GCP_PROJECT \
  --flatten="bindings[].members" \
  --filter="bindings.members:user@example.com"

# Cloud Run 実行ロールを付与
gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member=user:$(gcloud auth list --filter=status:ACTIVE --format="value(account)") \
  --role=roles/run.admin
```

---

## メモリプロファイリング

### Cloud Run ログから VRAM 使用量確認
```bash
gcloud run logs read vlm-agentic-rag-api --region=$GCP_REGION --limit=100 | grep -i memory

# 期待値：
# INFO - Loaded LLaVA 7B: ~8GB
# INFO - LoRA adapter: +0.1GB
# INFO - FAISS index: +0.5GB
# Total: ~8.6GB ✅ (16Gi 以内)
```

### 本番環境推奨スペック
```
┌─────────────────────────────┐
│  VLM + LoRA Agentic RAG     │
├─────────────────────────────┤
│ Memory: 16 Gi               │
│ CPU:     4 cores (x86_64)   │
│ Timeout: 3600 sec (1 hour)  │
│ Instances: 2 (concurrent)   │
└─────────────────────────────┘
```

---

## 本番環境への移行

### セキュリティ設定
```bash
# API 認証追加（本番）
gcloud run deploy vlm-agentic-rag-api \
  --image=$IMAGE_URI \
  --region=$GCP_REGION \
  --no-allow-unauthenticated \  # ← 認証を有効化
  --update-secrets="HF_TOKEN=projects/$GCP_PROJECT/secrets/hf-token:latest" \
  --project=$GCP_PROJECT

# HF Token 秘密管理
gcloud secrets create hf-token --data-file=- <<< "your_hf_token_here"
```

### カスタムドメイン設定
```bash
# Cloud Run + Cloud Load Balancer で カスタムドメイン設定
# 詳細：https://cloud.google.com/run/docs/quickstarts/build-and-deploy

gcloud run services update-traffic vlm-agentic-rag-api \
  --to-revisions LATEST=100 \
  --region=$GCP_REGION
```

---

## 次のステップ

✅ **完了時チェックリスト**
- [ ] GCP プロジェクト作成・有効化
- [ ] Docker イメージをArtifact Registry にプッシュ
- [ ] Cloud Run サービスデプロイ
- [ ] /health エンドポイント確認（200 OK）
- [ ] /analyze と /search テスト実行
- [ ] ログ監視設定
- [ ] 本番セキュリティ設定

---

## コスト試算（月間）

| 項目 | 用途 | 月額（概算） |
|---|---|---|
| Cloud Run | API ホスティング | $10-30 |
| Artifact Registry | Docker イメージ | $0.10 |
| Cloud Storage | モデルキャッシュ | $1-5 |
| **合計** | | **$11-36** |

※ 無料枠：月100万リクエスト、40万 GB-秒まで無料

---

## リファレンス

- [Cloud Run ドキュメント](https://cloud.google.com/run/docs)
- [Artifact Registry](https://cloud.google.com/artifact-registry)
- [gcloud run deploy](https://cloud.google.com/sdk/gcloud/reference/run/deploy)
