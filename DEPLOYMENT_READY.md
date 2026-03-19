# Week 3 本番デプロイメント - 実行ガイド

## ✅ 検証結果

```
必須ファイル確認
✅ api_production.py (10,434 bytes)
✅ vlm_agentic_rag_complete.py (14,899 bytes)
✅ Dockerfile (2,142 bytes)
✅ requirements_production.txt (660 bytes)

LoRA 統合コード確認
✅ PeftModel インポート
✅ LoRA adapter ロード（HF from_pretrained）
✅ HF リポジトリ参照（Shion1124/vlm-lora-agentic-rag）
✅ lora_loaded フィールド追加
✅ lora_adapter メタデータ統合

本番依存パッケージ確認
✅ peft==0.8.0 （LoRA 必須）
✅ torch==2.2.2, transformers==4.38.0 （互換性確認済み）
✅ bitsandbytes==0.42.0 （4-bit 量子化）
✅ fastapi, uvicorn, pydantic （API フレームワーク）
✅ sentence-transformers, faiss-cpu （RAG）
✅ pdf2image, pillow （ドキュメント処理）
```

---

## 🚀 デプロイメント実行手順

### **方法 A: ローカル Docker テスト（GPU/CPU環境が必要）**

#### Step 1: Docker Desktop をインストール
```bash
# macOS
brew install --cask docker

# その後、Docker Desktop アプリを起動
```

#### Step 2: Docker イメージをビルド
```bash
cd /Users/yoshihisashinzaki/VLM

docker build -t vlm-agentic-rag:latest .
# 初回ビルド: 10-15分（LLaVA リポジトリダウンロード + Python 依存インストール）
```

#### Step 3: コンテナを起動
```bash
# GPU ありの場合
docker run --gpus all -p 8000:8000 vlm-agentic-rag:latest

# GPU なし（CPU のみ）
docker run -p 8000:8000 vlm-agentic-rag:latest

# バックグラウンド起動
docker run -d -p 8000:8000 --name vlm-api vlm-agentic-rag:latest
```

#### Step 4: API テスト

```bash
# ヘルスチェック
curl http://localhost:8000/health | jq .

# 期待値：
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "base_model": "llava-v1.5-7b",
#   "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
#   "lora_loaded": true    ← 重要: LoRA が実際にロードされている
# }
```

#### Step 5: テストスクリプト実行
```bash
chmod +x test_api.sh
./test_api.sh http://localhost:8000
```

---

### **方法 B: GCP Cloud Run デプロイ（推奨・クラウド）**

#### Step 1: Google Cloud SDK をインストール
```bash
# macOS
brew install google-cloud-sdk

# インストール後
gcloud init
# ブラウザが開く → ログイン → 許可
```

#### Step 2: GCP プロジェクト作成
```bash
# プロジェクト作成
gcloud projects create vlm-agentic-rag --set-as-default

# API 有効化
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

#### Step 3: Cloud Run デプロイ（自動ビルド）
```bash
cd /Users/yoshihisashinzaki/VLM

gcloud run deploy vlm-agentic-rag-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --set-env-vars="HF_HOME=/models,TRANSFORMERS_CACHE=/models"

# デプロイ完了後、以下が表示される：
# Service URL: https://vlm-agentic-rag-api-xxx.run.app
```

#### Step 4: 本番 API テスト

```bash
SERVICE_URL="https://vlm-agentic-rag-api-xxx.run.app"

# ヘルスチェック
curl -s $SERVICE_URL/health | jq .

# ドキュメント分析テスト
curl -X POST -F "file=@files/2026_3q_summary_jp.pdf" \
  $SERVICE_URL/analyze | jq .

# セマンティック検索テスト
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "売上", "top_k": 3}' \
  $SERVICE_URL/search | jq .
```

---

## 📊 実行パターン別ガイド

### **パターン 1: Docker をインストール済み**
```
1. docker build -t vlm-agentic-rag:latest .
2. docker run -p 8000:8000 vlm-agentic-rag:latest  
3. curl http://localhost:8000/health
```

### **パターン 2: gcloud をインストール済み**
```
1. gcloud projects create vlm-agentic-rag --set-as-default
2. gcloud services enable run.googleapis.com
3. gcloud run deploy vlm-agentic-rag-api --source . --region us-central1
```

### **パターン 3: 何もインストール済みでない場合**
```
1. brew install --cask docker
2. Docker Desktop アプリを起動
3. docker build -t vlm-agentic-rag:latest .
4. docker run -p 8000:8000 vlm-agentic-rag:latest
```

---

## 📁 ファイル構成（デプロイ用）

```
/Users/yoshihisashinzaki/VLM/
├── api_production.py              ✅ FastAPI サーバー（LoRA 統合済み）
├── vlm_agentic_rag_complete.py    ✅ VLM + LoRA パイプライン
├── Dockerfile                     ✅ Docker イメージ定義
├── docker-compose.yml             ✅ Docker Compose（オプション）
├── requirements_production.txt     ✅ Python 依存パッケージ
├── test_api.sh                    ✅ API テストスクリプト
├── deploy.sh                      ✅ デプロイメント準備スクリプト
└── vlm_agentic_rag_colab.ipynb    ✅ Colab 学習ノートブック（参考）
```

---

## 🔧 トラブルシューティング

### 問題: Docker ビルド失敗「LLaVA リポジトリダウンロード中に接続エラー」
```bash
# 解決: ネットワーク再接続 + リトライ
docker build --no-cache -t vlm-agentic-rag:latest .
```

### 問題: `curl: (7) Failed to connect to localhost:8000`
```bash
# 解決: コンテナが起動しているか確認
docker ps

# ログ確認
docker logs <container_id>

# コンテナを再起動
docker restart <container_id>
```

### 問題: Cloud Run デプロイ失敗「メモリ不足」
```bash
# 解決: メモリを増やす
gcloud run deploy vlm-agentic-rag-api \
  --image=... \
  --memory 32Gi \  # 32GB に増加
  --update
```

### 問題: `lora_loaded: false` になっている
```bash
# 原因: HF adapter ダウンロード失敗
# 解決: 以下を確認
1. インターネット接続確認
2. HF Token が必要な場合は環境変数設定
   export HF_TOKEN=<your_token>
3. ログで詳細確認
   gcloud run logs read vlm-agentic-rag-api --limit=50
```

---

## 📈 パフォーマンス目標値

```
┌──────────────────────────────────────┐
│   VLM + LoRA Agentic RAG            │
├──────────────────────────────────────┤
│ ヘルスチェック応答       50ms         │
│ PDF 1ページ 分析時間    2-3s         │
│ セマンティック検索      100-500ms    │
│ メモリ使用量           ~9.5GB (RAM) │
│ GPU メモリ             8GB (T4)      │
│ スループット           10-15 doc/min │
└──────────────────────────────────────┘
```

---

## ✅ デプロイメント完了チェックリスト

- [ ] ファイル検証（すべて ✅ 確認済み）
- [ ] LoRA 統合コード確認（すべて ✅ 確認済み）
- [ ] Docker インストール（または gcloud）
- [ ] Docker ビルド成功
- [ ] `/health` エンドポイント → `lora_loaded: true`
- [ ] `/analyze` エンドポイント → PDF 処理動作確認
- [ ] `/search` エンドポイント → セマンティック検索動作確認
- [ ] API ドキュメント（Swagger UI）アクセス可能（`/docs`）
- [ ] 本番 API URL を Stockmark 提出資料に記載

---

## 💼 次のステップ（Week 3 完了後）

1. **本番監視設定**
   - Cloud Logging でエラー監視
   - Cloud Alerts で異常検知

2. **セキュリティ強化**
   - API 認証を有効化（本番環境）
   - HF Token を Secret Manager で管理

3. **GitHub リポジトリ公開**
   - README.md 最終版執筆
   - デモ動画撮影
   - ライセンス設定（Apache 2.0）

4. **Stockmark 提出資料**
   - 本番 API URL を記載
   - パフォーマンス指標を提示
   - アーキテクチャ図を追加

---

**準備完了！** 👌 これで本番デプロイメント可能な状態です。

お疲れ様です 🎉
