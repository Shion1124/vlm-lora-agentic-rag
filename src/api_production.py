# ============================================================
# FastAPI: VLM + LoRA + Visual RAG + Agentic RAG 本番API
# 
# Usage:
#   uvicorn api_production:app --host 0.0.0.0 --port 8000 --reload
# 
# Test:
#   curl -X POST -F "file=@document.pdf" http://localhost:8000/analyze
#   curl -X POST -H "Content-Type: application/json" \
#        -d '{"query": "売上?" }' http://localhost:8000/multimodal-search
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import torch
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# ============================================================
# ロギング設定
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FastApp初期化
# ============================================================

app = FastAPI(
    title="VLM + LoRA + Visual RAG + Agentic RAG API",
    description="Multimodal document structuring: Visual RAG (image search) + Agentic RAG (text search)",
    version="2.0.0"
)

# ============================================================
# CORS設定
# ============================================================

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://*.run.app,http://localhost:3000,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ============================================================
# API Key 認証
# ============================================================

API_KEY = os.environ.get("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key 検証（API_KEY 環境変数が設定されている場合のみ有効）"""
    if not API_KEY:
        return  # API_KEY 未設定時は認証スキップ（開発用）
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# ============================================================
# グローバル変数
# ============================================================

model = None
tokenizer = None
image_processor = None
pipeline = None
vlm_loaded = False

# ============================================================
# Pydantic モデル
# ============================================================

class AnalyzeResponse(BaseModel):
    status: str
    filename: str
    pages_analyzed: int
    confidence_avg: float
    documents: list
    metadata: dict

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResponse(BaseModel):
    query: str
    results: list
    iterations: int
    strategies_used: list

class MultimodalSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class MultimodalSearchResponse(BaseModel):
    query: str
    multimodal_search: dict
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    base_model: str
    lora_adapter: str
    lora_loaded: bool
    visual_rag_available: bool
    timestamp: str

# ============================================================
# Startup: モデルロード
# ============================================================

@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルをロード"""
    global model, tokenizer, image_processor, pipeline, vlm_loaded
    
    logger.info("Starting server... Loading models")
    
    try:
        # LLaVA ロード試行
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from vlm_agentic_rag_complete import DocumentStructuringPipeline
        from peft import PeftModel
        
        model_id = "liuhaotian/llava-v1.5-7b"
        adapter_id = "Shion1124/vlm-lora-agentic-rag"
        model_name = get_model_name_from_path(model_id)
        
        logger.info(f"Loading base model: {model_id}")
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_id,
            None,
            model_name=model_name,
            load_4bit=True,
            device_map="auto"
        )
        
        # LoRA アダプターをロード
        try:
            logger.info(f"Loading LoRA adapter: {adapter_id}")
            model = PeftModel.from_pretrained(
                model,
                adapter_id,
                torch_dtype="auto"
            )
            logger.info("✅ LoRA adapter loaded successfully")
        except Exception as adapter_error:
            logger.warning(f"⚠️ LoRA adapter loading failed: {adapter_error}")
            logger.warning("Continuing with base model only (non-fine-tuned)")
        
        vlm_loaded = True
        logger.info("✅ LLaVA model with LoRA loaded successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ LLaVA loading failed: {e}")
        logger.warning("Continuing in mock mode")
        vlm_loaded = False
    
    # パイプライン初期化
    try:
        from vlm_agentic_rag_complete import DocumentStructuringPipeline
        pipeline = DocumentStructuringPipeline()
        if vlm_loaded:
            pipeline.vlm.model = model
            pipeline.vlm.tokenizer = tokenizer
            pipeline.vlm.image_processor = image_processor
            pipeline.vlm._loaded = True
            logger.info("✅ VLM integrated into pipeline (with LoRA fine-tuning)")
        try:
            logger.info("Initializing multimodal RAG (Visual + Agentic)...")
            pipeline.visual_rag.setup_clip()  # Visual RAG 初期化
            logger.info("✅ Pipeline initialized (multimodal ready)")
        except Exception as rag_error:
            logger.warning(f"⚠️ Pipeline RAG initialization failed: {rag_error}")
            logger.warning("Continuing with limited functionality")
    except Exception as e:
        logger.error(f"Pipeline initialization error: {e}")
        logger.warning("Server continuing in degraded mode (health check only)")

# ============================================================
# エンドポイント: /analyze
# ============================================================

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(file: UploadFile = File(...), _: None = Depends(verify_api_key)):
    """
    PDFまたは画像ファイルを分析して構造化JSON出力
    
    Args:
        file: PDF or image file
    
    Returns:
        AnalyzeResponse: 分析結果
    
    Example:
        curl -X POST -F "file=@report.pdf" http://localhost:8000/analyze
    """
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # ファイル検証
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )
    
    try:
        # ファイル一時保存
        temp_dir = Path("/tmp/uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"Processing file: {file.filename}")
        
        # VLM分析
        results = pipeline.process_document(str(temp_path))
        
        # 統計計算
        confidence_scores = [d.get("confidence", 0.5) for d in results]
        confidence_avg = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        logger.info(f"✅ Analysis complete: {len(results)} pages, confidence: {confidence_avg:.2%}")
        
        return AnalyzeResponse(
            status="success",
            filename=file.filename,
            pages_analyzed=len(results),
            confidence_avg=confidence_avg,
            documents=results,
            metadata={
                "base_model": "llava-v1.5-7b",
                "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
                "lora_loaded": vlm_loaded,
                "method": "VLM + LoRA Adaptation + Agentic RAG",
                "timestamp": datetime.now().isoformat(),
                "vlm_loaded": vlm_loaded
            }
        )
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ============================================================
# エンドポイント: /search
# ============================================================

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, _: None = Depends(verify_api_key)):
    """
    Agentic RAGで文書を検索
    
    Args:
        request: SearchRequest (query + top_k)
    
    Returns:
        SearchResponse: 検索結果
    
    Example:
        curl -X POST -H "Content-Type: application/json" \\
             -d '{"query": "売上は？"}' \\
             http://localhost:8000/search
    """
    
    if not pipeline or len(pipeline.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload documents first."
        )
    
    try:
        logger.info(f"Searching: {request.query}")
        
        result = pipeline.search(request.query)
        
        logger.info(f"Found {len(result['results'])} results in {result['iterations']} iterations")
        
        return SearchResponse(
            query=request.query,
            results=result['results'][:request.top_k],
            iterations=result['iterations'],
            strategies_used=result['strategies_used']
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ============================================================
# エンドポイント: /multimodal-search
# ============================================================

@app.post("/multimodal-search", response_model=MultimodalSearchResponse)
async def multimodal_search_documents(request: MultimodalSearchRequest, _: None = Depends(verify_api_key)):
    """
    Visual RAG + Agentic RAG マルチモーダル検索
    
    画像の視覚検索 + テキストの意味検索を統合
    
    Args:
        request: MultimodalSearchRequest (query + top_k)
    
    Returns:
        MultimodalSearchResponse: 統合検索結果
    
    Example:
        curl -X POST -H "Content-Type: application/json" \\
             -d '{"query": "売上グラフ", "top_k": 5}' \\
             http://localhost:8000/multimodal-search
    """
    
    if not pipeline or len(pipeline.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload documents first."
        )
    
    try:
        logger.info(f"Multimodal search: {request.query}")
        
        result = pipeline.multimodal_search(request.query, top_k=request.top_k)
        
        text_count = len(result['multimodal_search']['text_results']['items'])
        visual_count = len(result['multimodal_search']['visual_results']['items'])
        logger.info(f"Multimodal results: {text_count} text + {visual_count} visual")
        
        return MultimodalSearchResponse(
            query=request.query,
            multimodal_search=result['multimodal_search'],
            metadata=result['metadata']
        )
    
    except Exception as e:
        logger.error(f"Multimodal search error: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {str(e)}")

# ============================================================
# エンドポイント: /health
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    ヘルスチェック
    
    Returns:
        HealthResponse: サーバー状態（LoRA、Visual RAG 対応有無含む）
    """
    visual_rag_ready = pipeline and pipeline.visual_rag and pipeline.visual_rag.clip_model is not None
    
    return HealthResponse(
        status="healthy",
        model_loaded=vlm_loaded,
        base_model="llava-v1.5-7b",
        lora_adapter="Shion1124/vlm-lora-agentic-rag",
        lora_loaded=vlm_loaded,
        visual_rag_available=visual_rag_ready,
        timestamp=datetime.now().isoformat()
    )

# ============================================================
# エンドポイント: /docs
# ============================================================

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "VLM + LoRA + Visual RAG + Agentic RAG API",
        "version": "2.0.0",
        "architecture": "Multimodal (Visual + Text) RAG",
        "endpoints": {
            "/docs": "Swagger UI documentation",
            "/redoc": "ReDoc documentation",
            "/health": "Health check (includes Visual RAG status)",
            "/analyze": "POST Document analysis (with LoRA fine-tuning)",
            "/search": "POST Text search (Agentic RAG only)",
            "/multimodal-search": "POST Multimodal search (Visual RAG + Agentic RAG)"
        },
        "model_status": "loaded_with_lora" if vlm_loaded else "fallback_mode",
        "base_model": "llava-v1.5-7b",
        "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
        "visual_rag": "CLIP (OpenAI) for image search",
        "text_rag": "Sentence Transformers (all-MiniLM-L6-v2) for semantic search"
    }

# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    # ローカル開発用
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
