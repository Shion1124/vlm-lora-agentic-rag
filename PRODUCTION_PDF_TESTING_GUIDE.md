# 📋 本番環境でのPDF処理テスト完全ガイド

**対象**: Week 2-3 でLoRA学習完了後、実PDFで動作確認する手順

---

## 🎯 概要

このドキュメントは、実LoRA学習済みモデルを使って **実PDFファイルを構造化する方法** を段階的に説明します。

### フェーズ分け

| フェーズ | 時期 | 状態 | 主な作業 |
|--------|------|------|--------|
| **デモ** | Week 1 | 現在 | サンプルドキュメント + HF公開 |
| **実装準備** | Week 2 | 開発中 | FastAPI開発・テスト環境構築 |
| **検証** | Week 2-3 | テスト | 実PDFで動作確認 |
| **本番** | Week 3+ | 運用 | クラウドデプロイ |

---

## 📌 前提条件

本番環境でPDFを処理する前に、以下が完了していることを確認してください：

```
✅ 実LoRA学習が完了している
   └─ adapter_config.json 
   └─ adapter_model.safetensors
   └─ 学習済みモデルが HuggingFace に公開されている

✅ FastAPI + Gradio 環境が準備されている
   └─ api.py が作成されている
   └─ requirements.txt に依存関係が記載されている

✅ テスト用のPDFファイル用意
   └─ 決算報告書 (推奨)
   └─ 技術仕様書
   └─ 報告書など
```

---

## 🚀 Phase 1: ローカル環境でのテスト

### Step 1: 学習済みモデルの確認

```bash
# 学習済みLoRAディレクトリの確認
ls -la /path/to/lora_output/

# 必須ファイルのチェック
必要なファイル:
  ├─ adapter_config.json         ✅ LoRA設定
  ├─ adapter_model.safetensors   ✅ 学習済み重み
  ├─ tokenizer_config.json       ✅ トークナイザー
  ├─ special_tokens_map.json     ✅ 特殊トークン
  └─ README.md                   ✅ モデルカード
```

### Step 2: FastAPI サーバー起動

```python
# api.py を実装（以下は概要）

from fastapi import FastAPI, UploadFile, File
from vlm_agentic_rag_complete import DocumentStructuringPipeline
import shutil

app = FastAPI()

# グローバル パイプラインインスタンス
pipeline = DocumentStructuringPipeline()

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """実PDFファイルを分析・構造化"""
    
    # ①ファイル保存
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # ②PDF処理（実LLaVA推論）
        results = pipeline.process_document(temp_path)
        
        # ③インデックス化
        pipeline.build_agentic_rag()
        
        return {
            "status": "success",
            "filename": file.filename,
            "documents_processed": len(results),
            "documents": results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/search")
async def search_documents(query: str):
    """構造化ドキュメント内で検索"""
    if len(pipeline.documents) == 0:
        return {"error": "No documents indexed. Please upload first."}
    
    result = pipeline.search(query)
    return result

@app.get("/status")
def get_status():
    """システムステータス確認"""
    return {
        "documents_indexed": len(pipeline.documents),
        "avg_confidence": pipeline.get_statistics()['average_confidence'],
        "search_history": len(pipeline.rag.search_history)
    }

# 起動コマンド：
# uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

```bash
# サーバー起動
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 出力例:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

### Step 3: Swagger UIで動作確認

ブラウザで以下にアクセス：
```
http://localhost:8000/docs
```

#### Swagger UI での実行手順：

**① /analyze エンドポイントをテスト**

1. **Try it out** ボタンをクリック
2. ファイル選択: 実PDF（例: `Sony_2024_Report.pdf`）を選択
3. **Execute** ボタンをクリック

期待される応答：
```json
{
  "status": "success",
  "filename": "Sony_2024_Report.pdf",
  "documents_processed": 45,
  "documents": [
    {
      "title": "Financial Summary",
      "summary": "Company financial overview for FY2024...",
      "key_data": [
        "Revenue: 8.5 trillion JPY",
        "Operating Profit: 950 billion JPY",
        "EPS: 1,200 JPY"
      ],
      "insights": "Strong profitability despite market challenges",
      "confidence": 0.94,
      "page_number": 1
    },
    ...
  ]
}
```

**② /search エンドポイントをテスト**

1. **Try it out** ボタンをクリック
2. query 入力: `"売上高の推移は？"`
3. **Execute** ボタンをクリック

期待される応答：
```json
{
  "query": "売上高の推移は？",
  "results": [
    {
      "title": "Financial Summary",
      "key_data": ["Revenue: 8.5 trillion JPY", ...],
      "confidence": 0.94,
      "page_number": 1
    }
  ],
  "iterations": 1,
  "strategies_used": ["keyword_search"],
  "metadata": {
    "document_count": 45,
    "embedding_model": "all-MiniLM-L6-v2"
  }
}
```

---

## 🔬 Phase 2: 詳細なテスト・検証

### テスト 1: PDF処理性能

```python
# test_pdf_processing.py

import time
from datetime import datetime
from vlm_agentic_rag_complete import DocumentStructuringPipeline

# テスト用PDFリスト
test_files = [
    "Sony_2024_Report.pdf",         # 決算報告書
    "Samsung_TechSpec.pdf",          # 技術仕様書
    "McKinsey_MarketAnalysis.pdf"    # 市場分析
]

pipeline = DocumentStructuringPipeline()
pipeline.vlm.load_model()  # 実LLaVAモデル読み込み

# テスト実行
results = {}

for pdf_file in test_files:
    print(f"\n📄 Testing: {pdf_file}")
    print("=" * 70)
    
    # 処理時間計測
    start_time = time.time()
    
    try:
        # PDF処理
        docs = pipeline.process_document(pdf_file)
        
        # RAGインデックス化
        pipeline.build_agentic_rag()
        
        elapsed_time = time.time() - start_time
        
        stats = pipeline.get_statistics()
        
        results[pdf_file] = {
            "status": "success",
            "documents": len(docs),
            "avg_confidence": stats['average_confidence'],
            "processing_time_sec": elapsed_time
        }
        
        print(f"✅ Success")
        print(f"   Documents: {len(docs)}")
        print(f"   Avg Confidence: {stats['average_confidence']:.2%}")
        print(f"   Processing Time: {elapsed_time:.2f}s")
        
    except Exception as e:
        results[pdf_file] = {
            "status": "error",
            "error": str(e)
        }
        print(f"❌ Error: {e}")

# 結果サマリー
print("\n" + "=" * 70)
print("📊 Test Summary")
print("=" * 70)

for file, result in results.items():
    status = "✅" if result["status"] == "success" else "❌"
    print(f"\n{status} {file}")
    if result["status"] == "success":
        print(f"   - Docs: {result['documents']}")
        print(f"   - Confidence: {result['avg_confidence']:.2%}")
        print(f"   - Time: {result['processing_time_sec']:.2f}s")
    else:
        print(f"   - Error: {result['error']}")
```

### テスト 2: 検索品質

```python
# test_search_quality.py

from vlm_agentic_rag_complete import DocumentStructuringPipeline

pipeline = DocumentStructuringPipeline()

# 実PDFを処理
pipeline.process_document("Sony_2024_Report.pdf")
pipeline.build_agentic_rag()

# テスト検索クエリ
test_queries = [
    "売上高は？",           # 基本質問
    "営業利益の推移",       # 複合質問
    "経営課題は何か？",     # 洞察質問
]

print("🔍 Search Quality Test")
print("=" * 70)

for query in test_queries:
    result = pipeline.search(query)
    
    print(f"\n📌 Query: '{query}'")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Strategies: {' → '.join(result['strategies_used'])}")
    print(f"   Results: {len(result['results'])}")
    
    if result['results']:
        top_result = result['results'][0]
        print(f"\n   Top Result:")
        print(f"   - Title: {top_result['title']}")
        print(f"   - Confidence: {top_result['confidence']:.0%}")
        print(f"   - Key Data: {', '.join(top_result['key_data'][:2])}")
```

### テスト 3: エラーハンドリング

```python
# test_error_handling.py

from vlm_agentic_rag_complete import DocumentStructuringPipeline

pipeline = DocumentStructuringPipeline()

# ケース 1: サポートされていないファイル形式
print("Test 1: Unsupported file format")
try:
    pipeline.process_document("file.txt")
except ValueError as e:
    print(f"✅ Correctly caught: {e}")

# ケース 2: 存在しないファイル
print("\nTest 2: Non-existent file")
try:
    pipeline.process_document("nonexistent.pdf")
except FileNotFoundError as e:
    print(f"✅ Correctly caught: {e}")

# ケース 3: 空の検索結果
print("\nTest 3: Empty search (no documents)")
pipeline.documents = []
result = pipeline.search("query")
if len(result['results']) == 0:
    print("✅ Correctly returned empty results")

# ケース 4: 破損したPDF
print("\nTest 4: Corrupted PDF")
try:
    # 不正なPDFをテスト
    with open("/tmp/corrupted.pdf", "wb") as f:
        f.write(b"not a real pdf")
    pipeline.process_document("/tmp/corrupted.pdf")
except Exception as e:
    print(f"✅ Correctly caught: {type(e).__name__}")
```

---

## ✅ テストチェックリスト

実際にPDFを処理する際に確認すべき項目：

### PDFファイル処理

- [ ] PDF が正常に読み込まれている
- [ ] ページ数が正しく認識されている
- [ ] 画像への変換が成功している（/tmp/pdf_images/）
- [ ] 各ページの `confidence` が 0.7 以上
- [ ] `key_data` に重要情報が含まれている

### LLaVA推論

- [ ] LLaVAモデルが実際に読み込まれている（mock ではない）
- [ ] 推論時間が許容範囲（1ページ あたり 3-10秒）
- [ ] 日本語・英語両方に対応している
- [ ] テーブル・図表を認識している

### Agentic RAG検索

- [ ] キーワード検索が動作している
- [ ] 意味検索（semantic search）が動作している
- [ ] 複数戦略での再検索が機能している
- [ ] 検索結果が正確である（confidence > 0.8）

### エラーハンドリング

- [ ] 無効なPDFでエラーが出ていない
- [ ] グレースフルに例外処理されている
- [ ] エラーメッセージが わかりやすい

---

## 📊 パフォーマンスベンチマーク

期待される処理時間：

| 処理 | 所要時間 |
|-----|--------|
| PDF → 画像化（10ページ） | 2-3秒 |
| LLaVA推論（10ページ） | 30-60秒（T4 GPU） |
| FAISSインデックス化 | 1-2秒 |
| Agentic RAG検索 | 0.5-1秒 |
| **トータル** | **35-65秒** |

---

## 🐛 トラブルシューティング

### 問題: LLaVAが読み込めない

```
Error: ModuleNotFoundError: No module named 'llava'
```

**解決策:**
```bash
pip install git+https://github.com/haotian-liu/LLaVA.git
pip install transformers bitsandbytes accelerate peft
```

### 問題: GPU メモリ不足

```
RuntimeError: CUDA out of memory
```

**解決策:**
```python
# 4-bit量子化を有効化
pipeline.vlm = VLMHandler(use_4bit=True)
```

### 問題: PDF変換エラー

```
Error: AttributeError: module 'pdf2image' has no attribute 'convert_from_path'
```

**解決策:**
```bash
pip install pdf2image
# poppler をインストール（macOS）
brew install poppler
```

### 問題: FAISSインデックスエラー

```
Error: IndexFlatL2 error
```

**解決策:**
```python
# embedding dimension の確認
embeddings = pipeline.rag.embeddings
print(f"Shape: {embeddings.shape}")  # (num_docs, 384)

# FAISSインデックスの再作成
pipeline.rag.index_documents(pipeline.documents)
```

---

## 📈 本番運用のポイント

### キャッシング戦略

```python
# 同一PDFの再処理をキャッシュ
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_processing(file_hash: str) -> List[Dict]:
    return pipeline.process_document(file_path)
```

### スケーリング対応

```python
# 非同期処理で複数PDFを同時処理
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_multiple_pdfs(pdf_files: List[str]):
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [
            asyncio.to_thread(pipeline.process_document, pdf)
            for pdf in pdf_files
        ]
        results = await asyncio.gather(*tasks)
    return results
```

### モニタリング

```python
# ログ出力
import logging

logger = logging.getLogger(__name__)

logger.info(f"Processing: {filename}")
logger.info(f"Documents: {len(results)}")
logger.info(f"Confidence: {avg_conf:.2%}")
```

---

## 🚀 本番デプロイ

### Docker化

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### クラウドデプロイ (Google Cloud Run)

```bash
# イメージビルド
gcloud builds submit --tag gcr.io/PROJECT_ID/vlm-agentic-rag

# デプロイ
gcloud run deploy vlm-agentic-rag \
  --image gcr.io/PROJECT_ID/vlm-agentic-rag \
  --platform managed \
  --region us-central1 \
  --memory 4Gi
```

---

**最終確認**: Swagger UI で全エンドポイントがテストできた場合、本番運用準備完了です！ ✅
