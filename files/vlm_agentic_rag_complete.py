"""
VLM + Agentic RAG Pipeline for Document Structuring
====================================================
ストックマーク求人対応：Visual RAG PoC
- LLaVA（Vision-Language Model）
- Agentic RAG（自律検索・検証ループ）
- PDF → 構造化JSON → 検索可能
"""

# ============================================================
# ① 環境構築・初期化
# ============================================================

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# ====== Colab environment setup ======
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    print("🔧 Colab detected. Setting up environment...")
    os.system("pip install -q pdf2image pillow")
    os.system("pip install -q sentence-transformers faiss-cpu")
    os.system("pip install -q gradio")
    os.system("pip install -q peft bitsandbytes accelerate")


# ============================================================
# ② LLaVA モデル管理クラス
# ============================================================

class VLMHandler:
    """LLaVAモデル管理＋推論"""
    
    def __init__(self, model_id: str = "liuhaotian/llava-v1.5-7b", use_4bit: bool = True):
        self.model_id = model_id
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """モデルロード（軽量版：T4対応）"""
        print(f"📦 Loading VLM: {self.model_id}")
        
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            
            model_name = get_model_name_from_path(self.model_id)
            
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_id,
                None,
                model_name=model_name,
                load_4bit=self.use_4bit,
                device_map="auto"
            )
            print("✅ VLM loaded successfully")
            
        except Exception as e:
            print(f"⚠️  LLaVA import failed. Using fallback (疑似VLM)...")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """フォールバック：疑似VLM（デモ用）"""
        self.model = None
        print("📝 Using mock VLM for demonstration")
    
    def analyze_image(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        画像を構造化JSONに変換
        
        Args:
            image_path: 画像ファイルパス
            prompt: カスタムプロンプト（省略可）
        
        Returns:
            構造化辞書
        """
        if prompt is None:
            prompt = """
            この画像を解析して以下のJSON形式で出力してください:
            {
              "title": "タイトル",
              "summary": "要約",
              "key_data": ["重要データ1", "重要データ2"],
              "insights": "洞察",
              "confidence": 0.8
            }
            """
        
        # Mock processing（本番環境ではLLaVA使用）
        result = {
            "title": Path(image_path).stem,
            "summary": "This is a structured analysis of the document page.",
            "key_data": ["データ抽出例1", "データ抽出例2"],
            "insights": "自動構造化によるビジネス洞察",
            "confidence": 0.85,
            "source": image_path
        }
        
        return result


# ============================================================
# ③ Agentic RAG エンジン
# ============================================================

class AgenticRAGEngine:
    """
    Agentic RAG: 自律的検索・検証ループ
    
    ユースケース：
    - 複数の検索戦略を動的に選択
    - 結果を検証し、不足なら再検索
    - 複雑なマルチステップ推論に対応
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.documents = []
        self.embeddings = None
        self.index = None
        self.search_history = []
        
    def setup_embedder(self):
        """埋め込みモデルのセットアップ"""
        from sentence_transformers import SentenceTransformer
        
        print(f"📚 Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        print("✅ Embedder ready")
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """ドキュメント群をインデックス化"""
        import faiss
        
        self.documents = documents
        
        # ドキュメントをテキスト化
        doc_texts = [json.dumps(doc, ensure_ascii=False) for doc in documents]
        
        # 埋め込みを計算
        print("🔍 Building FAISS index...")
        embeddings = self.embedder.encode(doc_texts, convert_to_numpy=True)
        
        # FAISSインデックス構築
        self.embeddings = embeddings
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        
        print(f"✅ Indexed {len(documents)} documents")
    
    def _search_by_keyword(self, query: str, k: int = 3) -> List[Dict]:
        """キーワード検索（精密）"""
        q_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_embedding.astype('float32'), k)
        return [self.documents[i] for i in indices[0]]
    
    def _search_by_semantic(self, query: str, k: int = 3) -> List[Dict]:
        """意味検索（拡張）"""
        # ここでは簡略化。実際は複数戦略の組み合わせ
        return self._search_by_keyword(query, k)
    
    def _verify_results(self, query: str, results: List[Dict]) -> bool:
        """結果検証：必要に応じて再検索"""
        # 単純な検証ロジック
        if len(results) == 0:
            return False
        
        # 結果に信頼度フィールドがあるか確認
        avg_confidence = np.mean([r.get("confidence", 1.0) for r in results])
        return avg_confidence > 0.5
    
    def agentic_search(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Agentic RAG: 自律検索ループ
        
        フロー：
        1. 初期検索（キーワード）
        2. 結果検証
        3. 不十分なら戦略切り替えて再検索
        4. 最終結果キュレーション
        """
        iteration = 0
        results = []
        strategy_log = []
        
        while iteration < max_iterations:
            iteration += 1
            
            if iteration == 1:
                # 第1段階：キーワード検索
                strategy = "keyword_search"
                results = self._search_by_keyword(query, k=5)
            
            elif iteration == 2:
                # 第2段階：意味検索（拡張）
                strategy = "semantic_search"
                results = self._search_by_semantic(query, k=5)
            
            else:
                # 第3段階：ハイブリッド
                strategy = "hybrid_search"
                keyword_results = self._search_by_keyword(query, k=3)
                semantic_results = self._search_by_semantic(query, k=3)
                results = keyword_results + semantic_results
            
            strategy_log.append(strategy)
            
            # 結果検証
            is_valid = self._verify_results(query, results)
            
            if is_valid or iteration == max_iterations:
                break
        
        # 記録
        self.search_history.append({
            "query": query,
            "iterations": iteration,
            "strategies": strategy_log,
            "result_count": len(results)
        })
        
        return {
            "query": query,
            "results": results,
            "iterations": iteration,
            "strategies_used": strategy_log,
            "metadata": {
                "timestamp": str(np.datetime64('now')),
                "embedding_model": self.embedding_model_name,
                "document_count": len(self.documents)
            }
        }


# ============================================================
# ④ ドキュメント構造化パイプライン
# ============================================================

class DocumentStructuringPipeline:
    """
    VLM + Agentic RAG の統合パイプライン
    
    フロー：
    PDF → 画像化 → VLM分析 → 構造化JSON → Agentic RAGインデックス
    """
    
    def __init__(self):
        self.vlm = VLMHandler()
        self.rag = AgenticRAGEngine()
        self.documents = []
    
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """PDFを画像に変換"""
        try:
            from pdf2image import convert_from_path
            
            print(f"📄 Converting PDF: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=200)
            
            # 一時フォルダに保存
            temp_dir = Path("/tmp/pdf_images")
            temp_dir.mkdir(exist_ok=True)
            
            image_paths = []
            for i, img in enumerate(images):
                save_path = temp_dir / f"page_{i}.png"
                img.save(save_path)
                image_paths.append(str(save_path))
            
            print(f"✅ Converted {len(image_paths)} pages")
            return image_paths
        
        except Exception as e:
            print(f"⚠️  PDF conversion error: {e}")
            return []
    
    def process_document(self, file_path: str) -> List[Dict]:
        """
        ドキュメント処理：VLM + 構造化
        """
        # ファイルタイプ判定
        if file_path.endswith(".pdf"):
            image_paths = self.pdf_to_images(file_path)
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            image_paths = [file_path]
        else:
            raise ValueError("Unsupported file format")
        
        # VLMで各ページを分析
        results = []
        for img_path in image_paths:
            print(f"🔍 Analyzing: {img_path}")
            analysis = self.vlm.analyze_image(img_path)
            analysis["page_number"] = len(results) + 1
            results.append(analysis)
        
        self.documents = results
        return results
    
    def build_agentic_rag(self):
        """Agentic RAGのセットアップ"""
        self.rag.setup_embedder()
        self.rag.index_documents(self.documents)
    
    def search(self, query: str) -> Dict[str, Any]:
        """Agentic RAGで検索"""
        return self.rag.agentic_search(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報"""
        return {
            "total_documents": len(self.documents),
            "search_history": self.rag.search_history,
            "average_confidence": np.mean([d.get("confidence", 1.0) for d in self.documents])
        }


# ============================================================
# ⑤ Gradio UI
# ============================================================

def create_gradio_interface():
    """Gradio UIの作成"""
    import gradio as gr
    
    pipeline = DocumentStructuringPipeline()
    
    def upload_and_process(file):
        """ファイルアップロード＆処理"""
        if file is None:
            return "❌ ファイルを選択してください"
        
        file_path = file.name
        results = pipeline.process_document(file_path)
        pipeline.build_agentic_rag()
        
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    def search_documents(query: str):
        """検索実行"""
        if len(pipeline.documents) == 0:
            return "❌ まずドキュメントをアップロードしてください"
        
        result = pipeline.search(query)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def get_stats():
        """統計表示"""
        stats = pipeline.get_statistics()
        return json.dumps(stats, ensure_ascii=False, indent=2)
    
    # UI構築
    with gr.Blocks(title="VLM Agentic RAG") as demo:
        gr.Markdown("# 📊 VLM + Agentic RAG Document Structuring")
        gr.Markdown("PDFや画像をアップロード → VLMで構造化 → Agentic RAGで検索")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="📁 PDFまたは画像をアップロード")
                upload_btn = gr.Button("🚀 処理開始")
                upload_output = gr.Textbox(label="処理結果", lines=10)
            
            with gr.Column():
                query_input = gr.Textbox(label="🔍 検索クエリ")
                search_btn = gr.Button("🔎 検索実行")
                search_output = gr.Textbox(label="検索結果", lines=10)
        
        stats_btn = gr.Button("📈 統計情報")
        stats_output = gr.Textbox(label="統計", lines=5)
        
        # ボタンアクション
        upload_btn.click(upload_and_process, inputs=[file_input], outputs=[upload_output])
        search_btn.click(search_documents, inputs=[query_input], outputs=[search_output])
        stats_btn.click(get_stats, outputs=[stats_output])
    
    return demo


# ============================================================
# ⑥ メイン実行
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 VLM + Agentic RAG Pipeline")
    print("=" * 60)
    
    # CLI モード
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        pipeline = DocumentStructuringPipeline()
        
        results = pipeline.process_document(file_path)
        pipeline.build_agentic_rag()
        
        print("\n📄 Structured Documents:")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        
        # サンプル検索
        query = "売上トレンド" if len(sys.argv) < 3 else sys.argv[2]
        search_result = pipeline.search(query)
        print(f"\n🔍 Search Results for '{query}':")
        print(json.dumps(search_result, ensure_ascii=False, indent=2))
    
    # Gradio UI モード
    else:
        print("🌐 Launching Gradio UI...")
        demo = create_gradio_interface()
        demo.launch(share=True)
