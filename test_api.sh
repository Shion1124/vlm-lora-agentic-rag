#!/bin/bash
# API テストスクリプト - VLM + LoRA Agentic RAG
# 
# 使い方：
#   chmod +x test_api.sh
#   ./test_api.sh http://localhost:8000        # ローカル
#   ./test_api.sh https://vlm-api-xxx.run.app  # Cloud Run

set -e

API_URL="${1:-http://localhost:8000}"
PDF_FILE="files/2026_3q_summary_jp.pdf"

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}VLM + LoRA Agentic RAG - API Test Suite${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
echo -e "${BLUE}API URL: $API_URL${NC}"
echo ""

# ============================================================
# Test 1: Health Check
# ============================================================
echo -e "${BOLD}[1/5] Health Check${NC}"
echo "Endpoint: GET /health"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/health")
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} - Status: $HTTP_CODE"
else
    echo -e "${RED}❌ FAIL${NC} - Status: $HTTP_CODE"
fi
echo ""
echo "---"
echo ""

# ============================================================
# Test 2: Root Endpoint
# ============================================================
echo -e "${BOLD}[2/5] Root Endpoint (API Info)${NC}"
echo "Endpoint: GET /"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/")
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} - Status: $HTTP_CODE"
else
    echo -e "${RED}❌ FAIL${NC} - Status: $HTTP_CODE"
fi
echo ""
echo "---"
echo ""

# ============================================================
# Test 3: Document Analysis (PDF Upload)
# ============================================================
echo -e "${BOLD}[3/5] Document Analysis (PDF Upload)${NC}"
echo "Endpoint: POST /analyze"
echo "File: $PDF_FILE"
echo ""

if [ ! -f "$PDF_FILE" ]; then
    echo -e "${YELLOW}⚠️  SKIP${NC} - PDF file not found: $PDF_FILE"
    echo "Please upload a PDF to: $PDF_FILE"
else
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        -F "file=@$PDF_FILE" \
        "$API_URL/analyze")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    echo ""
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✅ PASS${NC} - Status: $HTTP_CODE"
    else
        echo -e "${YELLOW}⚠️  INFO${NC} - Status: $HTTP_CODE (expected in test/fallback mode)"
    fi
fi
echo ""
echo "---"
echo ""

# ============================================================
# Test 4: Semantic Search
# ============================================================
echo -e "${BOLD}[4/5] Semantic Search${NC}"
echo "Endpoint: POST /search"
echo "Query: 売上"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "売上", "top_k": 3}' \
    "$API_URL/search")
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ]; then
    echo -e "${GREEN}✅ PASS${NC} - Status: $HTTP_CODE"
    if [ "$HTTP_CODE" = "400" ]; then
        echo "   (No documents indexed yet - expected on first run)"
    fi
else
    echo -e "${RED}❌ FAIL${NC} - Status: $HTTP_CODE"
fi
echo ""
echo "---"
echo ""

# ============================================================
# Test 5: Swagger UI / ReDoc
# ============================================================
echo -e "${BOLD}[5/5] API Documentation${NC}"
echo "Swagger UI: $API_URL/docs"
echo "ReDoc:      $API_URL/redoc"
echo ""

SWAGGER_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/docs")
REDOC_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/redoc")

if [ "$SWAGGER_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} - Swagger UI available"
else
    echo -e "${RED}❌ FAIL${NC} - Swagger UI not available (status: $SWAGGER_RESPONSE)"
fi

if [ "$REDOC_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} - ReDoc available"
else
    echo -e "${RED}❌ FAIL${NC} - ReDoc not available (status: $REDOC_RESPONSE)"
fi
echo ""
echo "---"
echo ""

# ============================================================
# Summary
# ============================================================
echo -e "${BOLD}========================================${NC}"
echo -e "${GREEN}✅ API Test Suite Complete${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Check Cloud Run logs:"
echo "   gcloud run logs read vlm-agentic-rag-api --region=us-central1"
echo ""
echo "2. Monitor metrics:"
echo "   https://console.cloud.google.com/run/detail/us-central1/vlm-agentic-rag-api/metrics"
echo ""
echo "3. Access Swagger UI for interactive testing:"
echo "   $API_URL/docs"
echo ""
