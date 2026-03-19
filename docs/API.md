# API Reference

## Base URL

```
https://vlm-agentic-rag-api-744279114226.us-central1.run.app
```

## Authentication

Currently **unauthenticated** (public API). For production, add authentication via:
- API Key
- JWT Bearer Token
- OAuth2

## Endpoints

### 1. Root / API Info

**GET** `/`

Returns API metadata and available endpoints.

**Response:**
```json
{
  "name": "VLM + LoRA Agentic RAG API",
  "version": "1.0.0",
  "endpoints": {
    "/docs": "Swagger UI documentation",
    "/redoc": "ReDoc documentation",
    "/health": "Health check endpoint",
    "/analyze": "POST - Document analysis (with LoRA fine-tuning)",
    "/search": "POST - Search documents (Agentic RAG)"
  },
  "model_status": "fallback_mode",
  "base_model": "llava-v1.5-7b",
  "lora_adapter": "Shion1124/vlm-lora-agentic-rag"
}
```

### 2. Health Check

**GET** `/health`

Checks API health and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": false,
  "base_model": "llava-v1.5-7b",
  "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
  "lora_loaded": false,
  "timestamp": "2026-03-19T22:37:36.983644"
}
```

**Fields:**
- `status`: "healthy" | "degraded" | "unhealthy"
- `model_loaded`: Boolean - whether VLM is loaded
- `base_model`: Model identifier
- `lora_adapter`: LoRA adapter identifier
- `lora_loaded`: Boolean - whether LoRA is integrated
- `timestamp`: ISO 8601 timestamp

### 3. Document Analysis

**POST** `/analyze`

Analyzes uploaded documents (PDF or image) using LoRA-fine-tuned VLM.

**Request:**
```bash
curl -X POST \
  -F "file=@document.pdf" \
  https://vlm-agentic-rag-api-744279114226.us-central1.run.app/analyze
```

**Parameters:**
- `file` (multipart/form-data, required): PDF or image file
  - Supported: `.pdf`, `.png`, `.jpg`, `.jpeg`
  - Max size: 10MB

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "pages_analyzed": 5,
  "confidence_avg": 0.87,
  "documents": [
    {
      "page": 0,
      "title": "Document Title",
      "summary": "Summary of content...",
      "key_data": ["data1", "data2"],
      "insights": "Key insights...",
      "confidence": 0.91,
      "source": "document.pdf"
    }
  ],
  "metadata": {
    "processing_time_ms": 5420,
    "model_used": "llava-v1.5-7b + LoRA"
  }
}
```

**Status Codes:**
- `200 OK`: Analysis successful
- `400 Bad Request`: Invalid file format
- `413 Payload Too Large`: File exceeds size limit
- `503 Service Unavailable`: Model not loaded

### 4. Semantic Search (Agentic RAG)

**POST** `/search`

Searches indexed documents using multi-strategy agentic RAG.

**Request:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search term",
    "top_k": 3
  }' \
  https://vlm-agentic-rag-api-744279114226.us-central1.run.app/search
```

**Parameters:**
```json
{
  "query": "string (required) - search query",
  "top_k": "integer (optional, default: 3) - number of results"
}
```

**Response:**
```json
{
  "query": "search term",
  "results": [
    {
      "title": "Result Title",
      "content": "Relevant content snippet...",
      "score": 0.92,
      "source": "document_name.pdf",
      "page": 2
    }
  ],
  "iterations": 2,
  "strategies_used": ["keyword_search", "semantic_search"],
  "metadata": {
    "search_time_ms": 234,
    "documents_indexed": 20,
    "embedding_model": "all-MiniLM-L6-v2"
  }
}
```

**Agentic RAG Behavior:**
1. **Iteration 1**: Keyword search
2. **Iteration 2**: Semantic search
3. **Iteration 3**: Hybrid search (if needed)
4. **Verification**: Validates result relevance
5. **Refinement**: Repeats if confidence < 0.5

### 5. Documentation

**GET** `/docs`

Interactive Swagger UI for testing endpoints.

```
https://vlm-agentic-rag-api-744279114226.us-central1.run.app/docs
```

**GET** `/redoc`

ReDoc documentation format.

```
https://vlm-agentic-rag-api-744279114226.us-central1.run.app/redoc
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Request successful
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Endpoint not found
- `413 Payload Too Large`: Request body too large
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

- Currently **unlimited** for testing
- Production: Implement rate limiting (e.g., 100 req/min per IP)

## CORS

Currently allows all origins (`*`). For production, restrict to specific domains:

```python
allow_origins=["https://yourdomain.com"]
```

## Examples

### Python (requests library)

```python
import requests

BASE_URL = "https://vlm-agentic-rag-api-744279114226.us-central1.run.app"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/analyze", files=files)
    print(response.json())

# Search
query = {"query": "keyword", "top_k": 5}
response = requests.post(f"{BASE_URL}/search", json=query)
print(response.json())
```

### JavaScript (fetch API)

```javascript
const BASE_URL = "https://vlm-agentic-rag-api-744279114226.us-central1.run.app";

// Health check
fetch(`${BASE_URL}/health`)
  .then(r => r.json())
  .then(data => console.log(data));

// Search
fetch(`${BASE_URL}/search`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "keyword", top_k: 3 })
})
  .then(r => r.json())
  .then(data => console.log(data));
```

### cURL

```bash
# Health check
curl -s https://vlm-agentic-rag-api-744279114226.us-central1.run.app/health | jq .

# Search
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"search", "top_k":3}' \
  https://vlm-agentic-rag-api-744279114226.us-central1.run.app/search | jq .
```

## Notes

- **Model Loading**: First request may take 30-60 seconds while models load
- **Fallback Mode**: Without GPU, runs in mock mode (structure + metadata only)
- **GPU Support**: To enable full VLM + LoRA inference, deploy with GPU tier

