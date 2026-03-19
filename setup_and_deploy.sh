#!/bin/bash
# ============================================================
# VLM + LoRA - 本番デプロイメント インストール＆セットアップスクリプト
# ============================================================

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}================================================================${NC}"
echo -e "${BOLD}VLM + LoRA Agentic RAG - 本番デプロイメント セットアップ${NC}"
echo -e "${BOLD}${BLUE}================================================================${NC}"
echo ""

# ============================================================
# Step 1: Docker インストール確認
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 1: Docker をインストール中...${NC}"
echo ""

if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker は既にインストールされています${NC}"
    docker --version
else
    echo -e "${BLUE}Installing Docker Desktop via Homebrew...${NC}"
    brew install --cask docker
    echo -e "${GREEN}✅ Docker Desktop がインストールされました${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  重要: Docker Desktop を手動で起動してください${NC}"
    echo "Applications フォルダから Docker.app を開いてください"
    echo ""
fi

echo ""

# ============================================================
# Step 2: Google Cloud SDK インストール
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 2: Google Cloud SDK をインストール中...${NC}"
echo ""

if command -v gcloud &> /dev/null; then
    echo -e "${GREEN}✅ gcloud は既にインストールされています${NC}"
    gcloud --version | head -1
else
    echo -e "${BLUE}Installing Google Cloud SDK via Homebrew...${NC}"
    brew install google-cloud-sdk
    echo -e "${GREEN}✅ Google Cloud SDK がインストールされました${NC}"
fi

echo ""

# ============================================================
# Step 3: gcloud 初期化
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 3: gcloud 初期化（認証）${NC}"
echo ""
echo -e "${BLUE}以下のコマンドを実行して、ブラウザで Google アカウントにログインしてください:${NC}"
echo ""
echo -e "${BOLD}gcloud auth login${NC}"
echo ""
echo -e "${YELLOW}このスクリプトが自動で gcloud init を実行します。${NC}"
echo ""

# gcloud init
if command -v gcloud &> /dev/null; then
    gcloud init --skip-diagnostics
    echo -e "${GREEN}✅ gcloud が初期化されました${NC}"
fi

echo ""

# ============================================================
# Step 4: GCP プロジェクト作成
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 4: GCP プロジェクト作成${NC}"
echo ""

PROJECT_ID="vlm-agentic-rag"
echo -e "${BLUE}プロジェクト ID: $PROJECT_ID${NC}"

# プロジェクト作成（既存の場合はスキップ）
if gcloud projects describe $PROJECT_ID &> /dev/null; then
    echo -e "${GREEN}✅ プロジェクト $PROJECT_ID は既に存在します${NC}"
    gcloud config set project $PROJECT_ID
else
    echo -e "${BLUE}プロジェクに $PROJECT_ID を作成中...${NC}"
    gcloud projects create $PROJECT_ID --name="VLM + LoRA Agentic RAG"
    gcloud config set project $PROJECT_ID
    echo -e "${GREEN}✅ プロジェクト $PROJECT_ID が作成されました${NC}"
fi

echo ""

# ============================================================
# Step 5: API 有効化
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 5: Google Cloud API を有効化${NC}"
echo ""

APIS="run.googleapis.com artifactregistry.googleapis.com storage-api.googleapis.com"

for api in $APIS; do
    echo -e "${BLUE}Enabling $api...${NC}"
    gcloud services enable $api --quiet
done

echo -e "${GREEN}✅ API が有効化されました${NC}"
echo ""

# ============================================================
# Step 6: Cloud Run デプロイメント
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 6: Cloud Run デプロイメント準備${NC}"
echo ""

echo -e "${BOLD}デプロイメント実行コマンド:${NC}"
echo ""
echo -e "${BLUE}cd /Users/yoshihisashinzaki/VLM${NC}"
echo ""
echo -e "${BLUE}gcloud run deploy vlm-agentic-rag-api \\${NC}"
echo -e "${BLUE}  --source . \\${NC}"
echo -e "${BLUE}  --region us-central1 \\${NC}"
echo -e "${BLUE}  --platform managed \\${NC}"
echo -e "${BLUE}  --allow-unauthenticated \\${NC}"
echo -e "${BLUE}  --memory 16Gi \\${NC}"
echo -e "${BLUE}  --cpu 4 \\${NC}"
echo -e "${BLUE}  --timeout 3600${NC}"
echo ""

# ============================================================
# Step 7: デプロイメント実際実行
# ============================================================

echo -e "${BOLD}${YELLOW}▶ Step 7: Cloud Run へデプロイ実行${NC}"
echo ""

if [ -d "/Users/yoshihisashinzaki/VLM" ]; then
    cd /Users/yoshihisashinzaki/VLM
    
    echo -e "${BLUE}デプロイを開始しています（これには数分かかります）...${NC}"
    echo ""
    
    gcloud run deploy vlm-agentic-rag-api \
      --source . \
      --region us-central1 \
      --platform managed \
      --allow-unauthenticated \
      --memory 16Gi \
      --cpu 4 \
      --timeout 3600 \
      --quiet
    
    echo ""
    echo -e "${GREEN}✅ Cloud Run デプロイメント完了！${NC}"
    echo ""
    
    # ============================================================
    # Step 8: サービス URL 取得
    # ============================================================
    
    echo -e "${BOLD}${YELLOW}▶ Step 8: サービス URL 取得${NC}"
    echo ""
    
    SERVICE_URL=$(gcloud run services describe vlm-agentic-rag-api \
      --region us-central1 \
      --format='value(status.url)')
    
    echo -e "${GREEN}✅ API URL: ${BOLD}$SERVICE_URL${NC}"
    echo ""
    
    # ============================================================
    # Step 9: API テスト
    # ============================================================
    
    echo -e "${BOLD}${YELLOW}▶ Step 9: API テスト実行${NC}"
    echo ""
    
    echo -e "${BLUE}▶ ヘルスチェック:${NC}"
    curl -s "$SERVICE_URL/health" | jq . 2>/dev/null || echo "JSON parse error - response displayed as-is"
    echo ""
    
    echo -e "${BLUE}▶ API Info:${NC}"
    curl -s "$SERVICE_URL/" | jq . 2>/dev/null || echo "JSON parse error - response displayed as-is"
    echo ""
    
    echo -e "${BOLD}${YELLOW}▶ Step 10: テストスクリプト実行${NC}"
    echo ""
    
    if [ -f "test_api.sh" ]; then
        chmod +x test_api.sh
        ./test_api.sh "$SERVICE_URL"
    else
        echo -e "${YELLOW}⚠️  test_api.sh が見つかりません${NC}"
    fi
    
else
    echo -e "${RED}❌ /Users/yoshihisashinzaki/VLM ディレクトリが見つかりません${NC}"
    exit 1
fi

echo ""

# ============================================================
# 完了メッセージ
# ============================================================

echo -e "${BOLD}${GREEN}================================================================${NC}"
echo -e "${BOLD}${GREEN}✅ デプロイメント完了！${NC}"
echo -e "${BOLD}${GREEN}================================================================${NC}"
echo ""

echo -e "${BOLD}本番 API エンドポイント:${NC}"
echo -e "${BOLD}${BLUE}$SERVICE_URL${NC}"
echo ""

echo -e "${BOLD}次のステップ:${NC}"
echo "  1. Swagger UI で API テスト: $SERVICE_URL/docs"
echo "  2. ログ確認: gcloud run logs read vlm-agentic-rag-api --limit=50"
echo "  3. メトリクス確認: Cloud Console でモニタリング"
echo ""

echo -e "${BOLD}Stockmark 提出資料に記載:${NC}"
echo "  API Endpoint: $SERVICE_URL"
echo "  LoRA Adapter: Shion1124/vlm-lora-agentic-rag"
echo ""
