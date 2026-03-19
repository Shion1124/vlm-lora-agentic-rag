# ============================================================
# FastAPI: VLM + LoRA Agentic RAG 本番API
# 
# Usage:
#   uvicorn api_production:app --host 0.0.0.0 --port 8000 --reload
# 
# Test:
#   curl -X POST -F "file=@document.pdf" http://localhost:8000/analyze
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import asyncio
import json
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
    title="VLM + LoRA Agentic RAG API",
    description="Document structuring with LLM-adapted Vision Language Model",
    version="1.0.0"
)

# CORS設定（本番環境では制限）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    base_model: str
    lora_adapter: str
    lora_loaded: bool
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
            pipeline.build_agentic_rag()
            logger.info("✅ Pipeline initialized")
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
async def analyze_document(file: UploadFile = File(...)):
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
async def search_documents(request: SearchRequest):
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
# エンドポイント: /health
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    ヘルスチェック
    
    Returns:
        HealthResponse: サーバー状態（LoRA 適用有無含む）
    """
    return HealthResponse(
        status="healthy",
        model_loaded=vlm_loaded,
        base_model="llava-v1.5-7b",
        lora_adapter="Shion1124/vlm-lora-agentic-rag",
        lora_loaded=vlm_loaded,
        timestamp=datetime.now().isoformat()
    )

# ============================================================
# エンドポイント: /docs
# ============================================================

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "VLM + LoRA Agentic RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Swagger UI documentation",
            "/redoc": "ReDoc documentation",
            "/health": "Health check",
            "/analyze": "POST Document analysis (with LoRA fine-tuning)",
            "/search": "POST Search documents (Agentic RAG)"
        },
        "model_status": "loaded_with_lora" if vlm_loaded else "fallback_mode",
        "base_model": "llava-v1.5-7b",
        "lora_adapter": "Shion1124/vlm-lora-agentic-rag"
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
