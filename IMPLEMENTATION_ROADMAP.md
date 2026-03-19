# 🚀 VLM Agentic RAG ポートフォリオ - 完全実装ロードマップ

**対象**: ストックマーク「AI R&Dエンジニア（VLM・マルチモーダル）」求人応募  
**ゴール**: 4週間で本番環境対応ポートフォリオの完成→応募  
**開始日**: 本日（Day 1）  

---

## 📅 フェーズ概要

```
Week 1: PoC + HuggingFace公開（このフェーズが最重要）
  ├─ Days 1-2: Colab Notebookで動作確認
  ├─ Days 3-4: HuggingFaceアップロード実行
  └─ Days 5-7: GitHub公開・デモリンク取得

Week 2: API本番化 + ドキュメント
  ├─ Days 8-10: FastAPI開発・テスト
  ├─ Days 11-13: Docker化・キャッシング最適化
  └─ Days 14: API v1.0 リリース

Week 3: コミュニティ発信 + 面接対準備
  ├─ Days 15-17: ブログ執筆（Medium / Zenn）
  ├─ Days 18-20: YouTube動画制作（30秒デモ）
  └─ Days 21: 技術深掘り・面接Q&A完成

Week 4: 応募〜本面接
  ├─ Days 22-24: Stockmark応募対策
  ├─ Days 25-27: 書類選考・一次面接
  └─ Days 28-30: 技術面接・最終面接対策
```

---

## 🔥 **WEEK 1: PoC + HuggingFace公開（最優先）**

### **Days 1-2: Colab Notebook実行**

#### Day 1 のアクション

**朝（8:00-9:00）: 環境準備**
```
☐ Google Colab を新規作成
  https://colab.research.google.com/
  
☐ vlm_agentic_rag_colab.ipynb をダウンロード
  場所: /Users/yoshihisashinzaki/VLM/vlm_agentic_rag_colab.ipynb
  
☐ Colonadにアップロード（またはCopypaste）
  → Colab左側のファイルアイコン > Upload
```

**昼（11:00-13:00）: Cell実行**
```
☐ Cell 1: 環境セットアップ（pip install）
   → 所要時間: 3-5分
   → 出力:✅ Dependencies installed

☐ Cell 2-5: VLM + RAGエンジン初期化
   → 所要時間: 2分
   → 確認: Pipeline initialized successfully

☐ Cell 6-8: サンプル処理＆検索実行
   → 所要時間: 1分
   → 結果: 3つの検索結果が表示される
```

**夜（18:00-19:00）: HF統合テスト**
```
☐ Cell 9: README.md生成テスト
   → 確認: /tmp/lora_output/README.md が作成される
   → 内容: YAML frontmatter + モデルカード

☐ Cell 10: HFアップロード（dry-run）
   → トークン入力プロンプトが表示される（未入力でOK）
   → 処理: ファイル確認までを自動実行
```

#### Day 2 のアクション

**朝（8:00-10:00）: HuggingFace認証**

1️⃣ **HF トークン取得**
   ```
   Step A: https://huggingface.co/ にログイン
   
   Step B: 右上のアイコン → Settings (設定)
   
   Step C: 左メニュー → "Access Tokens"
   
   Step D: "+ Create new token" をクリック
          - Token name: "my-stockmark-upload"  
          - Token type: ✅ Write (重要！)
   
   Step E: Create token → コピーして安全に保管
   ```

2️⃣ **Colab Cell 10再実行**
   ```python
   # トークンを入力するプロンプトが表示される
   from huggingface_hub import login
   login()  # <- このセルで"huggingface_hub"と入力
   ```

**昼（11:00-13:00）: モデルカード確認**

```
☐ Cell 9の出力を確認
  - base_model ✅
  - datasets ✅  
  - training config ✅
  - license ✅

☐ 内容を修正（必要に応じて）
  # 例：モデル名を変更したい場合
  title_line = "YOUR-CUSTOM-NAME-HERE"
  # Cell 9を再実行
```

**夕方（15:00-17:00）: Gradio UI + 動作確認**

```
☐ Cell 11: Gradio UI起動
   出力例:
   🌐 Gradio URL: https://xxxxxxxx.gradio.live
   
☐ URLをブラウザでアクセス
  → 📊 UI画面が表示される
  → "Search" ボタンで検索テスト
  
☐ Share URLをコピーして保存
  （後でLinkedInに貼付用）
```

---

### **Days 3-4: HuggingFaceアップロード実行**

#### オプション A: 学習済みモデルがある場合

```python
# Cell 1: 既存モデルの準備
OUT_LORA_DIR = "/path/to/your/trained/lora"
# adapter_config.json
# adapter_model.safetensors  
# tokenizer files

# Cell 9: READMEを自動生成
# Cell 10: uploadを実行
```

#### オプション B: デモモード（現在）

```python
# Cell 10のアップロード部分
# 以下を実行：

#実際のアップロード（トークン入力後）
api.upload_folder(
    folder_path=str(STAGE_DIR),
    repo_id="your_username/vlm-qwen-lora",  # 自分の名前に変更
    repo_type="model",
    exist_ok=True
)

# 出力:
# ✅ Upload completed
# URL: https://huggingface.co/your_username/vlm-qwen-lora
```

---

### **Days 5-7: GitHub公開**

#### Day 5: リポジトリ初期化

```bash
# ローカルマシンで実行

# 1. 作業ディレクトリ作成
mkdir vlm-agentic-rag
cd vlm-agentic-rag

# 2. プロジェクトファイルをコピー
cp /Users/yoshihisashinzaki/VLM/files/* .
cp /Users/yoshihisashinzaki/VLM/vlm_agentic_rag_colab.ipynb .

# 3. Git初期化
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 4. 初期コミット
git add .
git commit -m "Initial commit: VLM Agentic RAG for Stockmark"
```

#### Day 6: GitHub リモート設定

```bash
# 1. GitHub でリポジトリ作成（ブラウザ）
#    https://github.com/new
#    repo name: vlm-agentic-rag
#    description: "Vision Language Model + Agentic RAG for document structuring"
#    Public ☑️

# 2. ローカルで push
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vlm-agentic-rag.git
git push -u origin main

# 3. 確認
# https://github.com/YOUR_USERNAME/vlm-agentic-rag にアクセス
# ファイルが表示される ✅
```

#### Day 7: Releases + README更新

```bash
# 1. Releaseをタグで作成
git tag -a v1.0.0 -m "Initial release: PoC complete"
git push origin v1.0.0

# 2. GitHub Releases ページで確認
# https://github.com/YOUR_USERNAME/vlm-agentic-rag/releases

# 3. README_PROJECT.md を最新化
# Gradio share URL を埋め込む
# HF model card URL を埋め込む
```

---

## 🎯 **WEEK 2: API本番化（Phase 2）**

### **Days 8-10: FastAPI開発**

#### 実装チェックリスト

```
☐ api.py の動作確認

  python api.py
  
  出力:
  INFO:  Uvicorn running on http://127.0.0.1:8000
  
☐ API Swagger UI確認
  http://localhost:8000/docs
  
☐ エンドポイント動作テスト

  POST /analyze
  POST /search
  GET /stats
```

#### キャッシング最適化

```python
# api.py に以下を追加

from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_search(query: str):
    # RAG検索結果をメモリキャッシュ
    return pipeline.search(query)
```

### **Days 11-13: Docker化**

```bash
# Dockerfile ビルド
docker build -t vlm-agentic-rag:latest .

# docker-compose で起動
docker-compose up -d

# 確認
curl http://localhost:8000/docs
```

### **Day 14: v1.0リリース**

```bash
git tag -a v1.0.0-api -m "API production ready"
git push origin v1.0.0-api
```

---

## 📝 **WEEK 3: コミュニティ発信 + 面接対策**

### **Days 15-17: ブログ執筆**

#### 投稿プラットフォーム

| プラットフォーム | タイトル案 | 所要時間 |
|--------------|----------|--------|
| **Zenn** | 「VLM + Agentic RAG で企業ドキュメント自動構造化」 | 3時間 |
| **Medium** | 「Building Production-Ready Multi-Modal RAG Systems」 | 3時間 |
| **Qiita** | 「LLaVA + FAISS で実装する自律検索」 | 2時間 |

#### ブログ構成

```
1. Introduction
   └─ 問題提起（PDFの複雑さ）
   └─ 従来RAGの限界

2. Architecture
   └─ VLM（LLaVA）の役割
   └─ Agentic RAGの仕組み
   └─ FAISS インデックス化

3. Implementation
   └─ Code walk-through
   └─ Python コード例集

4. Benchmark
   └─ 処理時間比較
   └─ 精度評価（Confidence scores）

5. Deployment
   └─ FastAPI
   └─ Docker運用
   └─ 本番環境のポイント
```

### **Days 18-20: YouTube動画制作**

#### デモ動画スクリプト（30秒）

```
[0:00-0:05] オープニング
  📊 VLM + Agentic RAG PoC のデモ

[0:05-0:15] 入力
  ❶ PDF/画像をアップロード
  ❷ LLaVAで自動構造化

[0:15-0:25] 処理
  ❸ Agentic RAGで多戦略検索
  ❹ 結果の検証・精緻化

[0:25-0:30] 結果
  ✅ JSON形式で自動出力
     + HuggingFaceで公開
```

### **Day 21: 面接Q&A完成**

深掘り50問（[HOUR INTERVIEW_MASTERCLASS.md](INTERVIEW_MASTERCLASS.md) からの抜粋）

```
Q1:  "VLMを選んだ理由"
     → ドメイン知識が必要（金融や技術仕様書）
     → 従来OCRでは捕捉できない図表・レイアウト

Q2:  "Agentic RAGの本質"
     → 単純な「retrieval」ではく「verification」ループ
     → 不十分な結果は自動的に戦略切り替え

Q3:  "本番化のボトルネック"
     → LLM推論時間（→キャッシング）
     → スケーラビリティ（→キュー・非同期）
```

---

## 💼 **WEEK 4: 応募〜本面接**

### **Days 22-24: 応募対策**

```
☐ ポートフォリオリンク集約
 
  自己紹介文テンプレート:
  ─────────────────────────────────
  "I built a production-ready VLM + Agentic RAG system
   for document structuring using LLaVA, FAISS, and FastAPI.
   
   - GitHub: https://github.com/YOUR_NAME/vlm-agentic-rag
   - HuggingFace: https://huggingface.co/YOUR_NAME/vlm-qwen-lora
   - Demo: [Gradio share URL]
   - Blog: [Zenn article]
  ─────────────────────────────────

☐ オンライン履歴書・ポートフォリオサイト
  https://notion.so または https://my-portfolio.com
```

### **Days 25-27: 面接本番対策**

```
☐ 技術面接シミュレーション（友人・メンター）
  - ホワイトボード設計：多言語ドキュメント対응
  - コード質問：RAG検索のランキング改善
  - トラブルシューティング：GPU メモリ最適化

☐ 実装デモの準備
  - Colab notebookをライブで実行
  - API swagger UIデモンストレーション
```

---

## ✅ 実行チェックリスト

### Week 1
- [ ] Colab Notebook Day 1-2実行完了
- [ ] HuggingFace README.md 自動生成確認
- [ ] HFアップロード実行（トークン入力）
- [ ] Gradio share URL取得
- [ ] GitHub リポジトリ作成・push完了
- [ ] Releases v1.0.0 タグ作成

### Week 2
- [ ] api.py 動作確認
- [ ] FastAPI Swagger UI確認
- [ ] Docker build成功
- [ ] キャッシング実装完了
- [ ] v1.0-api リリース完了

### Week 3
- [ ] Zenn/Medium 投稿1本以上
- [ ] YouTube 30秒デモ動画アップロード
- [ ] 面接Q&A50問完成
- [ ] LinkedIn投稿（技術記事シェア）

### Week 4
- [ ] Stockmark応募完了
- [ ] 一次面接対策完了
- [ ] 技術面接デモ準備完了

---

## 🎁 ボーナス: Timing図（アップロードのタイミング）

```
Day 1-2: Colab実行 → 動作確認 ✅
   ↓
Day 3-4: HuggingFace Upload ✅ (Colab最後のセル)
   ↓
Day 5-7: GitHub公開 ✅
   ↓
Week 2: API開発（並行） ✅
   ↓
Week 3: ブログ・動画発信 ✅
   ↓
Week 4: Stockmark応募 🎯

⭐️ 重要ポイント:
- HFアップロード は Day 3-4 に すぐに実行！
  （Wait for API = 時間ロス）
- API開発は Week 2以降で十分
- ポートフォリオ完成度 > API完成度
```

---

## 📚 参考資料リンク

| リソース | リンク | 用途 |
|---------|-------|------|
| **HuggingFace docs** | https://huggingface.co/docs | モデルカード作成 |
| **LLaVA** | https://github.com/haotian-liu/LLaVA | VLM実装 |
| **FAISS** | https://github.com/facebookresearch/faiss | インデックス化 |
| **FastAPI** | https://fastapi.tiangolo.com/ | API開発 |
| **Gradio** | https://www.gradio.app/ | デモUI |
| **Stockmark** | https://stockmark.co.jp/ | 応募先 |

---

**Good luck! 🚀**
