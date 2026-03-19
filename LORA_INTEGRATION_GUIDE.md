# HuggingFace LoRA モデル統合ガイド

## 📋 状況整理

### Week 2 で実施したこと
```
Colab Cell 10: LoRA 訓練
├─ ベースモデル: LLaVA 7B
├─ 訓練データ: 3000 サンプル（LLaVA-Instruct-150K）
├─ LoRA Config: r=64, alpha=128
├─ 結果: loss = 0.9691 ✅
└─ 出力: /tmp/lora_output/
    ├─ adapter_model.bin (24.5 MB)
    ├─ adapter_config.json
    └─ config.json

Colab Cell 11: HuggingFace アップロード
└─ リポジトリ: Shion1124/vlm-lora-agentic-rag ✅ 公開
```

### 問題点：アップロードされたもの
```
❌ ベースモデル（LLaVA）自体はアップロードされていない
❌ アップロードされたのは LoRA adapter weights のみ

✅ adapter_model.bin    （LoRA パラメータ）
✅ adapter_config.json  （LoRA 設定）
✅ README.md           （モデルカード）
```

### 従来の api_production.py の問題
```python
# ❌ 現状：ベースモデルのみをロード
model = load_pretrained_model("liuhaotian/llava-v1.5-7b")
# LoRA adapter を全く適用していない！

# ✅ 正しい方法：ベース + LoRA を合成
from peft import PeftModel
base_model = load_pretrained_model("liuhaotian/llava-v1.5-7b")
model = PeftModel.from_pretrained(base_model, "Shion1124/vlm-lora-agentic-rag")
```

---

## ✅ 修正内容

### 1. モデルロード処理の改善

**ファイル**: `api_production.py` - `startup_event()` 関数

```python
# ❌ 修正前：ベースモデルのみ
tokenizer, model, image_processor, _ = load_pretrained_model(...)

# ✅ 修正後：LoRA アダプター統合
tokenizer, model, image_processor, _ = load_pretrained_model(...)

# LoRA アダプターをロード
from peft import PeftModel
model = PeftModel.from_pretrained(
    model,
    "Shion1124/vlm-lora-agentic-rag",
    torch_dtype="auto"
)
```

**メリット**：
- Colab で訓練したモデルがそのまま本番環境で使用可能
- 微調整の効果が実際に反映される
- モデル管理が一元化される

---

### 2. レスポンスモデル改善

**ファイル**: `api_production.py` - `HealthResponse` Pydantic モデル

```python
# ❌ 修正前：base_model のみ
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    timestamp: str

# ✅ 修正後：LoRA 情報も含む
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    base_model: str
    lora_adapter: str
    lora_loaded: bool      # ← LoRA 適用有無を明示
    timestamp: str
```

**期待値**: API 応答例
```json
{
  "status": "healthy",
  "model_loaded": true,
  "base_model": "llava-v1.5-7b",
  "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
  "lora_loaded": true,     // ← LoRA が実際に適用されている
  "timestamp": "2026-03-20T..."
}
```

---

### 3. メタデータ（API レスポンス）改善

**変更箇所**: `/analyze` エンドポイントの `metadata` フィールド

```python
# ❌ 修正前
metadata = {
    "model": "llava-v1.5-7b",
    "method": "LoRA-adapted VLM",
    ...
}

# ✅ 修正後
metadata = {
    "base_model": "llava-v1.5-7b",
    "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
    "lora_loaded": true,
    "method": "VLM + LoRA Adaptation + Agentic RAG",
    ...
}
```

---

## 🔧  動作確認

### ローカルテスト
```bash
# 1. FastAPI サーバー起動
python api_production.py

# 2. LoRA ロード状況を確認
curl http://localhost:8000/health | jq .

# 期待値：
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "base_model": "llava-v1.5-7b",
#   "lora_adapter": "Shion1124/vlm-lora-agentic-rag",
#   "lora_loaded": true      ← 重要！
# }
```

### Cloud Run デプロイ後
```bash
SERVICE_URL="https://vlm-agentic-rag-api-xxx.run.app"

# LoRA 適用確認
curl -s $SERVICE_URL/health | jq .lora_loaded

# 期待値: true
```

---

## ❓ よくある質問

### Q1: `Loaded: False` でも本当に問題ないの？
**A**: **いいえ、問題大ありです**。

- `Loaded: False` = Mock VLM（実際の推論が動作していない）
- LLaVA が読み込み失敗している状態
- Colab では動いても、ローカル/Cloud Run では環境の違いでロード失敗することがある

**対策**:
- CPU のみ実行（遅い）
- メモリを増やす（Docker メモリ制限を変更）
- Docker イメージのセットアップを見直す

### Q2: LoRA adapter なしで base model のみ使用するのは？
**A**: **効果がない＋無駄**です。

- Colab で 3000 サンプルで訓練した効果が失われる
- 推論精度が低下する（~5-10% 悪化期待）
- HF アップロードの価値が無くなる

---

## 📊 Architecture 再掲

```
┌──────────────────────────────────────────┐
│         ユーザーのPDF入力              │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Layer 1: Vision Language Model          │
│  ├─ Base: LLaVA 7B                      │
│  ├─ 4-bit quantization (bitsandbytes)   │
│  └─ ✅ HuggingFace から自動DL可能       │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Layer 2: LoRA 微調整層                 │
│  ├─ Adapter: Shion1124/vlm-lora-...    │
│  ├─ Rank: 64, Alpha: 128               │
│  ├─ 訓練済み (loss: 0.9691)           │
│  └─ ✅ PeftModel で動的適用           │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  Layer 3: Agentic RAG Retrieval         │
│  ├─ 多戦略検索 (keyword + semantic)     │
│  ├─ 自己検証ループ                      │
│  └─ ✅ PDF データセットから検索          │
└────────────┬─────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│      出力: 構造化JSON                   │
│      {                                   │
│        "title": "...",                   │
│        "key_data": [...],                │
│        "confidence": 92%,                │
│        "lora_adapted": true   ← 重要   │
│      }                                   │
└──────────────────────────────────────────┘
```

---

## 🚀 次のステップ

### 1. Docker ビルド＆テスト
```bash
docker build -t vlm-agentic-rag:latest .
docker run -p 8000:8000 vlm-agentic-rag:latest

# LoRA 適用確認
curl http://localhost:8000/health | jq .lora_loaded
```

### 2. 本番 Cloud Run デプロイ
```bash
gcloud run deploy vlm-agentic-rag-api \
  --image=... \
  --memory=16Gi \
  --cpu=4
```

### 3. 本番 API テスト
```bash
./test_api.sh https://vlm-agentic-rag-api-xxx.run.app
```

---

## 📝 ipynb について

### Q: Colab ipynb ファイルって必要だった？

**A: 学習フェーズだけで本番では不要**

```
Week 2: ipynb → LoRA 訓練 → HF Upload
         ↓
         訓練済みモデル（adapter_model.bin）
              ↓
              保存

Week 3: 本番環境
        ↓
        api_production.py がこの adapter を自動DL
        ↓
        PeftModel でロード・統合
        ↓
        推論実行
```

**つまり**:
- ipynb は学習用（1回実行したら不要）
- 本番は HF リポジトリのモデルを使用（API が自動管理）
- ipynb をデプロイする必要はない ✅

---

## ✅ チェックリスト

- [x] api_production.py で PeftModel インポート
- [x] startup で HF adapter ロード
- [x] HealthResponse に lora_loaded フィールド追加
- [x] メタデータに base_model + lora_adapter 明示
- [x] ロード失敗時の graceful fallback
- [x] Docker イメージに peft 含める（requirements_production.txt 確認）

---

**更新日**: 2026-03-20
**バージョン**: v1.1 (LoRA 統合版)
