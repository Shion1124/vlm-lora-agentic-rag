#!/usr/bin/env python3
"""
FastAPI: VLM + LoRA Agentic RAG 本番API
Lightweight test version (without ML dependencies)

Usage:
  source venv_vlm/bin/activate
  python api_test.py

Then test with:
  curl http://localhost:8000/health
  curl http://localhost:8000/
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging

# ============================================================
# ロギング設定
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 初期化
# ============================================================

app = FastAPI(
    title="VLM + LoRA Agentic RAG API",
    description="Document structuring with Vision Language Model",
    version="1.0.0"
)

# ============================================================
# Pydantic レスポンスモデル
# ============================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    mode: str

class InfoResponse(BaseModel):
    name: str
    version: str
    endpoints: dict
    status: str

# ============================================================
# グローバル状態
# ============================================================

SERVER_STATUS = {
    "started": datetime.now().isoformat(),
    "mode": "test_mode (lightweight, FastAPI only)",
    "ml_loaded": False
}

# ============================================================
# Startup イベント
# ============================================================

@app.on_event("startup")
async def startup_event():
    """サーバー起動時の処理"""
    logger.info("🚀 FastAPI Server Starting...")
    logger.info(f"📍 Mode: {SERVER_STATUS['mode']}")
    logger.info("✅ Server ready at http://localhost:8000")
    logger.info("📚 API Docs: http://localhost:8000/docs")

# ============================================================
# ヘルスチェック
# ============================================================

@app.get("/", response_model=InfoResponse)
async def root():
    """ルートエンドポイント"""
    return InfoResponse(
        name="VLM + LoRA Agentic RAG API",
        version="1.0.0",
        endpoints={
            "/docs": "Swagger UI documentation",
            "/redoc": "ReDoc documentation",
            "/health": "Health check endpoint",
            "/analyze": "POST - Document analysis (not available in test mode)",
            "/search": "POST - Search documents (not available in test mode)"
        },
        status="healthy_test_mode"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェック"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        mode="test_mode (FastAPI only, ML disabled)"
    )

@app.post("/analyze")
async def analyze_document():
    """本番モードでのドキュメント分析（テストモード非対応）"""
    raise HTTPException(
        status_code=503,
        detail="ML models not loaded in test mode. Please use Docker or production deployment."
    )

@app.post("/search")
async def search_documents():
    """本番モードでのドキュメント検索（テストモード非対応）"""
    raise HTTPException(
        status_code=503,
        detail="ML models not loaded in test mode. Please use Docker or production deployment."
    )

# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═════════════════════════════════════════════════════════════════╗
║         VLM + LoRA + Agentic RAG FastAPI - Test Server         ║
╠═════════════════════════════════════════════════════════════════╣
║ Starting lightweight FastAPI server (ML features disabled)      ║
║                                                                 ║
║ Endpoints:                                                      ║
║   GET  http://localhost:8000/          - API info              ║
║   GET  http://localhost:8000/health    - Health check          ║
║   GET  http://localhost:8000/docs      - Swagger UI            ║
║   POST http://localhost:8000/analyze   - Document analysis     ║
║   POST http://localhost:8000/search    - Search documents      ║
║                                                                 ║
║ For production with ML models, use Docker:                     ║
║   docker-compose up --build                                    ║
╚═════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
