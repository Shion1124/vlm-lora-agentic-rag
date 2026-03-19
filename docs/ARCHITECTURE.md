# Architecture & Technical Design

## System Architecture

```
┌─────────────────────────────────────┐
│    VLM + LoRA + Agentic RAG        │
│          (3-Layer System)           │
├─────────────────────────────────────┤
│ Layer 1: Vision Language Model      │
│ • Base: LLaVA-v1.5-7b               │
│ • Quantization: 4-bit NF4           │
│ • Memory: ~7GB (quantized)          │
├─────────────────────────────────────┤
│ Layer 2: LoRA Fine-tuning           │
│ • Rank: 64                          │
│ • Alpha: 128                        │
│ • Adapter: Shion1124/vlm-lora-*     │
│ • Training Loss: 0.9691             │
├─────────────────────────────────────┤
│ Layer 3: Agentic RAG                │
│ • Embedding: all-MiniLM-L6-v2       │
│ • Index: FAISS                      │
│ • Search: Multi-strategy            │
└─────────────────────────────────────┘
```

## Component Interactions

```
[User Request]
       ↓
[FastAPI Endpoint] (api_production.py)
       ↓
   ┌─────────────────────────────┐
   │  DocumentStructuringPipeline │
   │  (vlm_agentic_rag_complete)  │
   └─────────────────────────────┘
       ↓
   ┌─────┬──────────┬────────────┐
   ↓     ↓          ↓            ↓
[VLMHandler] [AgenticRAGEngine] [Analysis]
   • LLaVA-7B   • FAISS Index    • Structured
   • 4-bit      • Embeddings     • JSON Output
   • LoRA adapt • Multi-search
```

## Data Flow

### Document Analysis (/analyze)
```
PDF/Image → pdf2image → VLM Analysis → Structured JSON → Response
```

### Semantic Search (/search)
```
Query → Embedding → FAISS Index Lookup → Multi-Strategy Search → Results
```

## Technology Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| Inference | ML Framework | PyTorch 2.2.2 |
| LLM | Transformer | transformers 4.38.0 |
| Fine-tuning | Parameter Efficient | PEFT 0.8.0 |
| Quantization | 4-bit | bitsandbytes 0.42.0 |
| Embeddings | Sentence | sentence-transformers 2.2.2 |
| Search | Vector DB | FAISS 1.7.4 |
| API | REST Framework | FastAPI 0.104.1 |
| Deployment | Serverless | Google Cloud Run |

## Key Design Decisions

1. **4-bit Quantization**: Reduces memory from 28GB → 7GB
2. **LoRA**: Only 0.1% trainable parameters (67MB vs 7GB)
3. **FAISS**: Fast vector similarity search for RAG
4. **Agentic RAG**: Iterative search refinement for better results
5. **Cloud Run**: Serverless, auto-scaling, no GPU management

## Performance Characteristics

- **Model Load Time**: ~15 seconds (cold start)
- **Inference Time**: ~2 seconds (single query)
- **Memory Usage**: 16GB (Cloud Run allocation)
- **Throughput**: 2-5 requests/second (depending on GPU availability)
