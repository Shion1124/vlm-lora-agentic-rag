# 🚀 VLM + LoRA Agentic RAG - 本番デプロイメント実行ガイド

## ステップバイステップ実行手順（手動）

### **ステップ 1: Docker Desktop インストール & 起動**

```bash
# ① Homebrew でインストール
brew install --cask docker

# ② インストール完了後、Applications フォルダから Docker.app を起動
# または Spotlight で検索: cmd + space → "docker" → Enter

# ③ Docker が起動したか確認
docker --version
# Docker version 20.10.x, build xxxxxxxx
```

---

### **ステップ 2: Google Cloud SDK インストール**

```bash
# ① Homebrew でインストール
brew install google-cloud-sdk

# ② インストール確認
gcloud --version
# Google Cloud SDK x.y.z
```

---

### **ステップ 3: gcloud 認証 & プロジェクト初期化**

```bash
# ① ログイン（ブラウザが開きます）
gcloud auth login

# ② プロジェクト初期化
gcloud init --skip-diagnostics

# 質問が表示される：
Pick configuration to use:
 [1] Re-initialize this configuration [default] with new settings 
 [2] Create a new configuration
Please enter your numeric choice:  
Please enter a value between 1 and 2: 1 を選択

次は自分のアカウントを選択

Pick cloud project to use: 
 [1] acoustic-mix-431303-q9
 [2] gen-lang-client-0667170507
 [3] my-project-3719-1704300112301
 [4] Enter a project ID
 [5] Create a new project
Please enter numeric choice or text value (must 
exactly match list item):  5 を選択

# 1. "Do you want to configure a default Compute Region and Zone?" → Y
# 2. "Which Google Cloud region would you like to use?" → us-central1
# 3. "Do you want to configure a default Compute Zone?" → N
```

---

### **ステップ 4: GCP プロジェクト作成 & セットアップ**

プロジェクトID=vlm-rag

```bash
# ① プロジェクト作成
gcloud projects create vlm-agentic-rag --name="vlm-agentic-rag-v2"

# プロジェクトが既に存在する場合はスキップ
# Error: (gcloud.projects.create) HttpError 409: The project ID has been disabled or deleted.
# → その場合はプロジェクト ID を変更（例：vlm-agentic-rag-v2）

# ② プロジェクトを設定
gcloud config set project vlm-agentic-rag

# ③ API を有効化
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage-api.googleapis.com

# ④ 有効化確認（さらに5-10分かかる場合あり）
gcloud services list --enabled | grep -E "run|artifactregistry"
```

---

### **ステップ 5: Cloud Run へデプロイ実行**

```bash
# ① ワークディレクトリに移動
cd /Users/yoshihisashinzaki/VLM

# ② デプロイ実行（初回: 10-15分）
gcloud run deploy vlm-agentic-rag-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600

# ③ 完了を待つ
# ...
# Service URL: https://vlm-agentic-rag-api-abc123xyz.run.app
# ✅ デプロイメント成功
```

---

### **ステップ 6: サービス URL 取得**

```bash
# サービス URL の確認
gcloud run services describe vlm-agentic-rag-api \
  --region us-central1 \
  --format='value(status.url)'

# 出力例:
# https://vlm-agentic-rag-api-abc123xyz.run.app
```

---

### **ステップ 7: API テスト実行**

```bash
# ① サービス URL を変数に設定
SERVICE_URL="https://vlm-agentic-rag-api-xxx.run.app"

# ② ヘルスチェック
curl -s $SERVICE_URL/health | jq .

# 期待値:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "base_model": "llava-v1.5-7b",
#   "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
#   "lora_loaded": true    ← 重要: LoRA が実際にロード
# }

# ③ API Info
curl -s $SERVICE_URL/ | jq .

# ④ テストスクリプト実行
cd /Users/yoshihisashinzaki/VLM
chmod +x test_api.sh
./test_api.sh $SERVICE_URL
```

---

### **ステップ 8: Swagger UI アクセス**

```bash
# ブラウザでアクセス
open https://vlm-agentic-rag-api-xxx.run.app/docs

# または手動で URL を入力:
# https://vlm-agentic-rag-api-xxx.run.app/docs
```

---

### **ステップ 9: ログ確認**

```bash
# 最新 50 行のログ
gcloud run logs read vlm-agentic-rag-api --limit=50

# VLM + LoRA ロード状況を確認:
gcloud run logs read vlm-agentic-rag-api --limit=50 | grep -E "Loading|loaded|LoRA"

# 期待値:
# 2026-03-20 10:30:00 Loading base model: liuhaotian/llava-v1.5-7b
# 2026-03-20 10:30:15 Loading LoRA adapter: Shion1124/vlm-lora-agentic-rag
# 2026-03-20 10:30:20 ✅ LoRA adapter loaded successfully
```

---

## 🎯 トラブルシューティング

### **問題: 「gcloud: command not found」**
```bash
# 解決:
brew install google-cloud-sdk
# または PATH を再設定
export PATH="/opt/homebrew/bin:$PATH"
```

### **問題: 「Cannot connect to Docker daemon」**
```bash
# 解決: Docker Desktop アプリを起動してください
# Applications → Docker.app をダブルクリック
# またはターミナルから:
open /Applications/Docker.app
```

### **問題: デプロイ失敗「Dockerfile not found」**
```bash
# 解決: 正しいディレクトリにいることを確認
cd /Users/yoshihisashinzaki/VLM
ls -la Dockerfile  # ファイルが存在することを確認
```

### **問題: 「lora_loaded: false」**
```bash
# ログで詳細を確認
gcloud run logs read vlm-agentic-rag-api --limit=100

# 原因の例：
# - HuggingFace ダウンロード失敗
# - ネットワーク接続問題
# - メモリ不足

# 対策:
# 1. インターネット接続を確認
# 2. メモリを増やす:
gcloud run deploy vlm-agentic-rag-api \
  --update \
  --memory 32Gi \
  --region us-central1
```

### **問題: タイムアウト「504 Gateway Timeout」**
```bash
# 解決: タイムアウト時間を延長
gcloud run deploy vlm-agentic-rag-api \
  --update \
  --timeout 7200 \
  --region us-central1
```

---

## 📊 デプロイメント完了後の確認チェックリスト

- [ ] Docker がインストールされている
- [ ] gcloud がインストールされている
- [ ] gcloud login 完了
- [ ] GCP プロジェクト作成完了
- [ ] API が有効化されている
- [ ] Cloud Run デプロイ完了
- [ ] `/health` エンドポイント → `lora_loaded: true`
- [ ] `/docs` → Swagger UI にアクセス可能
- [ ] Test PDF ですぐにテスト可能

---

## 💼 Stockmark 提出資料への記載内容

```
【本番 API エンドポイント】
https://vlm-agentic-rag-api-abc123xyz.run.app

【主要エンドポイント】
- GET  /health         ヘルスチェック
- POST /analyze        ドキュメント分析
- POST /search         セマンティック検索
- GET  /docs           Swagger UI

【LoRA モデル情報】
- HF Repository: Shion1124/vlm-lora-agentic-rag
- Base Model: liuhaotian/llava-v1.5-7b (4-bit quantized)
- LoRA Config: r=64, alpha=128
- Training Loss: 0.9691

【本番環境スペック】
- Memory: 16 GB
- CPU: 4 cores
- Timeout: 3600 seconds
- Region: us-central1
```

---

## 🎓 完成！

**すべてのファイルが揃い、本番デプロイメント可能な状態です。**

お疲れ様でした 🎉

推奨実行時間: 約 30-45分（初回デプロイ）
