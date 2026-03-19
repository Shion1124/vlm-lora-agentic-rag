#!/bin/bash
# ============================================================
# VLM + LoRA Agentic RAG - Week 3 デプロイメントスクリプト
# 
# 使い方:
#   chmod +x deploy.sh
#   ./deploy.sh
# ============================================================

set -e  # エラーで停止

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================
# ヘルパー関数
# ============================================================

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_step() {
    echo -e "${BOLD}${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ============================================================
# メイン処理
# ============================================================

print_header "VLM + LoRA Agentic RAG - デプロイメント準備"

echo -e "${BOLD}このスクリプトは以下を実行します:${NC}"
echo "  1. 環境チェック (Docker, gcloud, Python)"
echo "  2. ファイル検証"
echo "  3. デプロイメント計画 (ローカル vs Cloud)"
echo ""

# ============================================================
# Step 1: 環境チェック
# ============================================================

print_header "Step 1: 環境チェック"

# Python
print_step "Python バージョン確認"
python_version=$(python3 --version 2>&1)
print_info "$python_version"
print_success "Python OK"

# Docker
print_step "Docker インストール確認"
if command -v docker &> /dev/null; then
    docker_version=$(docker --version)
    print_info "$docker_version"
    print_success "Docker インストール済み"
    HAS_DOCKER=1
else
    print_error "Docker がインストールされていません"
    HAS_DOCKER=0
    echo ""
    echo -e "${YELLOW}【macOS での Docker インストール】${NC}"
    echo "  brew install --cask docker"
    echo ""
fi

# gcloud
print_step "Google Cloud SDK インストール確認"
if command -v gcloud &> /dev/null; then
    gcloud_version=$(gcloud --version 2>&1 | head -1)
    print_info "$gcloud_version"
    print_success "gcloud インストール済み"
    HAS_GCLOUD=1
else
    print_error "Google Cloud SDK がインストールされていません"
    HAS_GCLOUD=0
    echo ""
    echo -e "${YELLOW}【macOS での gcloud インストール】${NC}"
    echo "  brew install google-cloud-sdk"
    echo ""
fi

# ============================================================
# Step 2: ファイル検証
# ============================================================

print_header "Step 2: 本番ファイル検証"

files_to_check=(
    "api_production.py"
    "vlm_agentic_rag_complete.py"
    "Dockerfile"
    "docker-compose.yml"
    "requirements_production.txt"
)

for file in "${files_to_check[@]}"; do
    print_step "$file 確認"
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        print_info "存在（$size）"
        print_success "$file OK"
    else
        print_error "$file が見つかりません"
        exit 1
    fi
done

# ============================================================
# Step 3: デプロイメント計画提示
# ============================================================

print_header "Step 3: デプロイメントオプション"

echo ""
echo -e "${BOLD}【オプション A】ローカル Docker テスト${NC}"
if [ $HAS_DOCKER -eq 1 ]; then
    echo -e "${GREEN}✅ Docker インストール済み${NC}"
    echo ""
    echo "以下のコマンドを実行:"
    echo -e "${BLUE}  docker build -t vlm-agentic-rag:latest .${NC}"
    echo -e "${BLUE}  docker run -p 8000:8000 vlm-agentic-rag:latest${NC}"
    echo ""
    echo "テスト:"
    echo -e "${BLUE}  curl http://localhost:8000/health${NC}"
else
    echo -e "${YELLOW}⚠️  Docker がインストールされていません${NC}"
    echo ""
    echo "以下のコマンドでインストール:"
    echo -e "${BLUE}  brew install --cask docker${NC}"
    echo ""
    echo "インストール後、Docker Desktop アプリを起動してから再度実行してください"
fi

echo ""
echo -e "${BOLD}【オプション B】GCP Cloud Run デプロイ（推奨）${NC}"
if [ $HAS_GCLOUD -eq 1 ]; then
    echo -e "${GREEN}✅ gcloud インストール済み${NC}"
    echo ""
    echo "以下のコマンドを実行:"
    echo -e "${BLUE}  gcloud projects create vlm-agentic-rag --set-as-default${NC}"
    echo -e "${BLUE}  gcloud services enable run.googleapis.com${NC}"
    echo -e "${BLUE}  gcloud run deploy vlm-agentic-rag-api --source . --region us-central1${NC}"
else
    echo -e "${YELLOW}⚠️  gcloud がインストールされていません${NC}"
    echo ""
    echo "以下のコマンドでインストール:"
    echo -e "${BLUE}  brew install google-cloud-sdk${NC}"
    echo ""
    echo "インストール後、以下で認証:"
    echo -e "${BLUE}  gcloud auth login${NC}"
fi

echo ""
echo -e "${BOLD}【オプション C】Python スクリプトで検証（今すぐ実行可能）${NC}"
echo -e "${GREEN}✅ Python 環境を使用${NC}"
echo ""
echo "以下のコマンドを実行:"
echo -e "${BLUE}  python3 validate_implementation.py${NC}"
echo ""

# ============================================================
# Step 4: 検証スクリプト提示
# ============================================================

print_header "次のステップ"

if [ $HAS_DOCKER -eq 1 ]; then
    echo -e "${BOLD}ローカル Docker テスト:${NC}"
    echo "  1. docker build -t vlm-agentic-rag:latest ."
    echo "  2. docker run -p 8000:8000 vlm-agentic-rag:latest"
    echo "  3. curl http://localhost:8000/health"
    echo ""
fi

if [ $HAS_GCLOUD -eq 1 ]; then
    echo -e "${BOLD}GCP Cloud Run デプロイ:${NC}"
    echo "  1. gcloud auth login"
    echo "  2. gcloud projects create vlm-agentic-rag"
    echo "  3. gcloud run deploy vlm-agentic-rag-api --source . --region us-central1"
    echo ""
fi

echo -e "${BOLD}Python 検証（即座に実行可能）:${NC}"
echo "  python3 validate_implementation.py"
echo ""

# ============================================================
# 完了メッセージ
# ============================================================

print_header "環境チェック完了"

echo -e "${GREEN}✅ 本番ファイルがすべて揃っています${NC}"
echo ""
echo -e "${BOLD}推奨実行順序:${NC}"
echo ""

if [ $HAS_DOCKER -eq 0 ] && [ $HAS_GCLOUD -eq 0 ]; then
    echo "  1️⃣  Python 検証スクリプト実行"
    echo "  2️⃣  Docker または gcloud をインストール"
    echo "  3️⃣  本番デプロイ実行"
elif [ $HAS_DOCKER -eq 1 ]; then
    echo "  1️⃣  ローカル Docker テスト"
    echo "  2️⃣  gcloud インストール（未インストール）"
    echo "  3️⃣  GCP Cloud Run デプロイ"
else
    echo "  1️⃣  Python 検証スクリプト実行"
    echo "  2️⃣  Docker インストール"
    echo "  3️⃣  ローカル Docker テスト"
fi

echo ""
echo -e "${BOLD}詳細ガイド:${NC}"
echo "  - GCP_DEPLOYMENT_GUIDE.md"
echo "  - WEEK3_COMPLETION_GUIDE.md"
echo "  - LORA_INTEGRATION_GUIDE.md"
echo ""
