# 📅 完全準備プラン：30日間で内定を目指す

**対象求人**: ◯◯◯◯株式会社「AI R&Dエンジニア（VLM・マルチモーダル）」  
**開始日**: 本日（Day 1）  
**ゴール**: 応募 → 技術面接 → 内定（30日以内）  

---

## 🎯 30日間の全体戦略

```
Week 1: ポートフォリオ完成・公開 ✅ 完了
  └─ Day 1-3: Colab実行 + LoRA学習 + HFアップロード ✅
  └─ Day 4-6: GitHub・デモ動画公開 ✅
  └─ Day 7  : LinkedIn で最初の投稿

Week 2: FastAPI本番化・デプロイ ✅ 完了
  └─ Day 8-10: FastAPI v2.0.0 実装（APIキー認証・マルチモーダル対応）✅
  └─ Day 11-13: Docker コンテナ化 + Visual RAG 統合 ✅
  └─ Day 14  : Cloud Run デプロイ完了 ✅

Week 3: ドキュメント・発信本格化 ✅ 完了
  └─ Day 15-17: 技術ブログ 5 本執筆完了 ✅
  └─ Day 18-20: 面接対策 (深い質問への回答)
  └─ Day 21  : デモ動画・技術解説動画完成

Week 4: 応募・本面接
  └─ Day 22-24: Stockmark に応募
  └─ Day 25-27: 書類選考対応
  └─ Day 28-30: 技術面接準備・本面接

Goal: 内定ゲット 🎉
```

### 📌 実績サマリー（2026-03-23 時点）

| 項目 | 状況 | 詳細 |
|------|------|------|
| Colab 実行 | ✅ 完了 | 全 27 セル実行成功 |
| LoRA 学習 | ✅ 完了 | LLaVA-150K 3000件、Loss=0.969 |
| HuggingFace | ✅ 公開済 | [Shion1124/vlm-lora-agentic-rag](https://huggingface.co/Shion1124/vlm-lora-agentic-rag) |
| GitHub | ✅ 公開済 | リポジトリ公開・複数コミット |
| FastAPI | ✅ 完了 | v2.0.0（APIキー認証・CORS制限・マルチモーダル） |
| Visual RAG | ✅ 完了 | CLIP (openai/clip-vit-base-patch32) + FAISS |
| Agentic RAG | ✅ 完了 | Sentence-Transformers + FAISS + BM25 |
| Docker | ✅ 完了 | python:3.10-slim ベース（CPU/GPU 両対応） |
| Cloud Run | ✅ 稼働中 | `vlm-agentic-rag-api-00004-rb7`、4 CPU / 16GB RAM |
| 技術ブログ | ✅ 5本完了 | 概要・LoRA・Agentic RAG・FastAPI/Deploy・最適化 |
| 本番 URL | ✅ 稼働中 | `https://vlm-agentic-rag-api-744279114226.us-central1.run.app` |

---

## 📋 Week 1：ポートフォリオ完成・公開 ✅ 完了

> **実績**: Colab 全 27 セル実行成功、LoRA 学習完了（Loss=0.969）、HuggingFace 公開済み

### Day 1-3：Colab 実行 + LoRA学習 + HFアップロード

#### Day 1（水）- Colab 実行

**朝（1時間）**
- [ ] Google Colab を新規ノートブック作成
- [ ] `vlm_agentic_rag_colab.ipynb` をアップロード
- [ ] Cell 1-3: 環境構築を実行（LLaVA + LoRA依存関係）

**昼（2時間）**
- [ ] Cell 4-7: VLM + Agentic RAG デモ実行
  - VLMHandler がLLaVAを正しくロード
  - サンプル文書で検索デモを実行
- [ ] Gradio UIが起動することを確認
- [ ] share link を取得して記録

**夜（1時間）**
- [ ] スクリーンショット・出力ログを保存
- [ ] Gradio link を LinkedIn 下書きに保存

#### Day 2（木）- LoRA学習実行

**朝（1.5時間）**
- [ ] Cell 8-9: LoRA学習準備
  - Unsloth インストール
  - 学習データ確認
  - LoRA config (r=64, alpha=128) セットアップ

**昼（2.5時間）**
- [ ] Cell 9-10: 実LoRA学習実行【重要】
  - 1 epoch 学習開始（5-10分）
  - **adapter_model.bin が /tmp/lora_output に生成**
  - 学習済み重みの確認

**夜（30分）**
- [ ] 学習完了ログを確認
- [ ] LLaVA + LoRA アダプタの確認

#### Day 3（金）- HuggingFace アップロード

**朝（1時間）**
- [ ] HuggingFace トークン取得
  - https://huggingface.co/settings/tokens
  - Scope: "Write" を選択
- [ ] Colab でトークン入力プロンプトに対応

**昼（1.5時間）**
- [ ] Cell 10: HFアップロード実行【重要】
  - adapter_model.bin + adapter_config.json + README.md をアップロード
  - **https://huggingface.co/Shion1124/qwen3-4b-struct-lora**

**夜（1時間）**
- [ ] HFモデルカードが正常にアップロードされたか確認
- [ ] Gradio share link + HF repo link を保存

---

### Day 4-6：GitHub・Demo動画 公開

#### Day 4（土）

**朝（1.5時間）**
- [ ] GitHub アカウント作成（未作成の場合）
- [ ] SSH キー設定
- [ ] 初期リポジトリ作成

**昼（2時間）**
- [ ] ローカルで git 初期化
  ```bash
  git init vlm-agentic-rag
  cd vlm-agentic-rag
  cp /path/to/outputs/* .
  git add .
  git commit -m "Initial commit: VLM Agentic RAG"
  ```
- [ ] GitHub にプッシュ
- [ ] リポジトリが公開されていることを確認

**夜（30分）**
- [ ] リポジトリの表示確認
- [ ] README が正しく表示されているか検確

#### Day 5（日）

**朝（2時間）**
- [ ] HuggingFace アカウント作成（未作成の場合）
- [ ] Write トークンを取得
- [ ] HuggingFace CLI をセットアップ

**昼（2時間）**
- [ ] ダミーモデルファイルを作成（またはColab学習結果を使用）
- [ ] HuggingFace にアップロード
  ```bash
  huggingface-cli upload yourusername/vlm-agentic-rag-lora \
    adapter_model.safetensors \
    adapter_config.json \
    README_MODEL_CARD.md as README.md
  ```

**夜（30分）**
- [ ] HuggingFace に正しくアップロードされたか確認
- [ ] Model Card が表示されているか確認

#### Day 6（月）

**朝（1時間）**
- [ ] GitHub Releases を作成
  ```bash
  git tag -a v1.0.0 -m "Initial Release"
  git push origin v1.0.0
  ```

**昼（2時間）**
- [ ] 最初の調整・改善
  - README が読みにくい箇所を修正
  - コメントを追加
  - 依存関係を確認

**夜（30分）**
- [ ] Day 4-6 の成果をスクリーンショット保存
- [ ] 公開ポートフォリオのスタート完了を記録

---

### Day 7：最初の振り返り

**朝（30分）**
- [ ] Week 1 の成果を整理
- [ ] チェックリストを確認

**昼（1時間）**
- [ ] ブログ執筆の準備
  - TECHNICAL_BLOG.md の内容を読み込む
  - 執筆構成を計画

**夜（30分）**
- [ ] Week 2 の計画を確認
- [ ] 必要な準備物をリストアップ

---

## 📣 Week 2：FastAPI 本番化・Cloud Run デプロイ ✅ 完了

> **実績**: FastAPI v2.0.0（APIキー認証・マルチモーダル対応）、Visual RAG (CLIP) 統合、Cloud Run デプロイ成功
>
> **本番 URL**: `https://vlm-agentic-rag-api-744279114226.us-central1.run.app`

### Day 8-10：FastAPI v2.0.0 実装 + Visual RAG 統合 ✅

#### Day 8（火）✅ 完了

**朝（1.5時間）**
- [x] FastAPI v2.0.0 API 設計
- [x] APIキー認証（X-API-Key ヘッダー）実装

**昼（2.5時間）**
- [x] マルチモーダルエンドポイント実装
  - `POST /analyze` — VLM 画像解析
  - `POST /search` — テキスト検索（Agentic RAG）
  - `POST /multimodal-search` — 画像+テキスト検索（Visual RAG）
  - `GET /health` — ヘルスチェック
- [x] CORS 制限（環境変数ベース）

**夜（1時間）**
- [x] ローカルテスト完了
- [x] セキュリティ対策確認（非rootユーザー、APIキー認証）

#### Day 9（水）✅ 完了

**朝（1時間）**
- [x] Visual RAG エンジン実装（CLIP + FAISS）
- [x] openai/clip-vit-base-patch32 統合

**昼（2時間）**
- [x] Agentic RAG エンジン実装
  - Sentence-Transformers (all-MiniLM-L6-v2)
  - FAISS IndexFlatL2 + BM25 ハイブリッド検索

**夜（1時間）**
- [x] マルチモーダル検索パイプライン統合テスト
- [x] CPU/GPU 両対応のフォールバック設計

#### Day 10（木）✅ 完了

**朝（1時間）**
- [x] production 設定の最終調整
- [x] requirements_cloudrun.txt 整備

**昼（2時間）**
- [x] Dockerfile 作成（CPU版: python:3.10-slim）
  - libgl1 対応（Debian Trixie 互換）
  - 非 root ユーザー設定
  - マルチステージビルド

**夜（1時間）**
- [x] ローカル Docker ビルド・テスト完了

---

### Day 11-13：Cloud Run デプロイ + ブログ執筆 ✅

#### Day 11（金）✅ 完了

**朝（1時間）**
- [x] GCP プロジェクト設定（vlm-agentic-rag）
- [x] Artifact Registry 有効化

**昼（1時間）**
- [x] Cloud Run デプロイ実行（`--source .` 方式）
  ```
  gcloud run deploy vlm-agentic-rag-api \
    --source . --region us-central1 \
    --memory 16Gi --cpu 4 \
    --set-env-vars API_KEY=your-secret-key
  ```
- [x] ビルド成功・デプロイ完了

**夜（1時間）**
- [x] 全エンドポイントの動作確認
  - `GET /health` → `{"status": "healthy"}` ✅
  - `GET /` → API 情報 v2.0.0 ✅
  - `POST /search` → APIキー認証動作 ✅

#### Day 12（土）✅ 完了

**朝（1時間）**
- [x] 技術ブログ記事執筆開始
- [x] BLOG_ARTICLE_001: VLM + LoRA + Visual RAG + Agentic RAG 概要

**昼（2時間）**
- [x] BLOG_ARTICLE_002: LoRA fine-tuning 実装ガイド
- [x] BLOG_ARTICLE_003: Agentic RAG 詳細解説

**夜（1時間）**
- [x] BLOG_ARTICLE_004: FastAPI + Docker + Cloud Run デプロイ
- [x] BLOG_ARTICLE_005: パフォーマンス最適化

#### Day 13（日）✅ 完了

**朝（30分）**
- [x] 全ブログ記事に Visual RAG + Cloud Run 実績を反映
- [x] ポートフォリオサマリー更新

**昼（1時間）**
- [x] 全ドキュメントの整合性確認
- [x] 本番 URL・API バージョン・エンドポイント記述の統一

**夜（30分）**
- [x] Git commit + push

---

### Day 14：ドキュメント最終整備 ✅

#### Day 14（月）✅ 完了

**朝（1.5時間）**
- [x] 30DAY_MASTERPLAN.md を実績ベースで更新
- [x] 00_PORTFOLIO_SUMMARY.md を最新状態に更新

**昼（3時間）**
- [x] 全 5 本のブログ記事に以下を反映:
  - Visual RAG (CLIP) 統合の記述
  - API v2.0.0（APIキー認証・マルチモーダル対応）
  - 実際の Cloud Run デプロイ結果
  - 本番 URL・エンドポイント・セキュリティ

**夜（1時間）**
- [x] 最終動作確認（本番環境でのヘルスチェック）
- [x] GitHub に全更新をプッシュ

---

## 🧠 Week 3：面接対策集中

### Day 15-17：技術理解の深掘り

#### Day 15（火）

**朝（2時間）**
- [ ] VLM の基礎を深掘り
  - Vision Encoder の仕組み
  - LLM との統合方法
  - 推論フロー

**昼（2時間）**
- [ ] Agentic RAG の深掘り
  - 各検索戦略の詳細
  - 検証ロジック
  - 実装の工夫

**夜（1時間）**
- [ ] ノートにまとめ

#### Day 16（水）

**朝（2時間）**
- [ ] FastAPI・Docker の深掘り
- [ ] API 設計の意思決定

**昼（2時間）**
- [ ] 本番環境対応の詳細理解
- [ ] スケーリング・監視

**夜（1時間）**
- [ ] 質問への回答を用意

#### Day 17（木）

**朝（2時間）**
- [ ] 面接 Q&A を読み込む（INTERVIEW_MASTERCLASS.md）
- [ ] 自分の言葉で回答を作成

**昼（2時間）**
- [ ] 実装トレードオフを理解
- [ ] 改善案を 3 つ以上準備

**夜（1時間）**
- [ ] アーキテクチャ図を手書きで描く練習

---

### Day 18-20：面接シミュレーション

#### Day 18（金）

**朝（1.5時間）**
- [ ] YouTube で「技術面接」動画を視聴
- [ ] よくある失敗パターンを学習

**昼（2時間）**
- [ ] 一人で面接シミュレーション
  - 15 秒自己紹介を話す
  - 3 分技術説明を話す
  - 5 問の質問に答える

**夜（1時間）**
- [ ] 音声を録音・再生
- [ ] 改善点をノート

#### Day 19（土）

**朝（1時間）**
- [ ] メンターor友人に連絡
- [ ] 30 分の模擬面接をお願い

**昼（1.5時間）**
- [ ] 模擬面接実施
  - 質問に答える
  - フィードバック受け取る

**夜（1.5時間）**
- [ ] フィードバックを整理
- [ ] 改善

#### Day 20（日）

**朝（2時間）**
- [ ] 改善点を反映
- [ ] 再度一人でシミュレーション

**昼（1時間）**
- [ ] 逆質問を 5 つ以上準備

**夜（1時間）**
- [ ] Week 3 のまとめ

---

### Day 21：最終準備

#### Day 21（月）

**朝（1.5時間）**
- [ ] GitHub リポジトリを最終チェック
- [ ] README が完璧か確認

**昼（1.5時間）**
- [ ] ブログ・SNS 投稿を確認
- [ ] アクセス数・反応を記録

**夜（1時間）**
- [ ] 応募準備
  - 履歴書・職務経歴書の準備
  - ポートフォリオリンクを用意

---

## 📬 Week 4：応募・本面接

### Day 22-24：Stockmark に応募

#### Day 22（火）

**朝（1時間）**
- [ ] Stockmark 採用ページを確認
  - 求人情報を再度読む
  - 応募方法を確認

**昼（2時間）**
- [ ] 応募書類を作成
  - 職務経歴書
  - ポートフォリオ説明文

**夜（1時間）**
- [ ] 書類をレビュー
- [ ] 誤字脱字をチェック

#### Day 23（水）

**朝（1時間）**
- [ ] 応募メールを作成
  ```
  件名：VLM Agentic RAG ポートフォリオを含む
         「AI R&Dエンジニア」応募について
  
  本文：
  VLM（Vision-Language Model）と Agentic RAG を
  組み合わせたドキュメント理解システムを開発しました。
  
  GitHub: ...
  ブログ: ...
  
  本システムが Stockmark の Visual RAG 課題に対する
  一つのソリューションになると考えています。
  ```

**昼（30分）**
- [ ] 応募ボタンをクリック
- [ ] 送信完了を記録

**夜（30分）**
- [ ] 応募完了の連絡を待つ

#### Day 24（木）

**朝（30分）**
- [ ] 返信状況を確認
- [ ] 返信がなければ別ルートで問い合わせ検討

**昼（1時間）**
- [ ] 別プロジェクトへの応募も検討
- [ ] 複数社に応募して確率を上げる

**夜（30分）**
- [ ] Day 25 以降の準備

---

### Day 25-27：書類選考対応

#### Day 25（金）

**朝（1時間）**
- [ ] 返信メールを確認
- [ ] 書類選考合格通知があったか確認

**昼（2時間）**
- [ ] 面接前のプレゼン資料を作成
  - アーキテクチャ図（PowerPoint）
  - デモ動画（MP4）

**夜（1時間）**
- [ ] 最終確認

#### Day 26（土）

**朝（2時間）**
- [ ] 面接対策：最終復習
  - Q&A をもう一度読む
  - 回答を話す練習

**昼（2時間）**
- [ ] 環境確認
  - Zoom・Meet の接続テスト
  - マイク・カメラ確認
  - 照明・背景確認

**夜（1時間）**
- [ ] 十分な睡眠

#### Day 27（日）

**朝（1時間）**
- [ ] 最終調整
- [ ] 資料を再度チェック

**昼（2時間）**
- [ ] リラックス
- [ ] 軽い運動で心身をリセット

**夜（1時間）**
- [ ] 十分な睡眠

---

### Day 28-30：技術面接準備・本面接

#### Day 28（月）

**朝（1.5時間）**
- [ ] 技術面接スケジュール確認
- [ ] 面接官の情報を調べる（LinkedIn など）

**昼（1.5時間）**
- [ ] 最終リハーサル
  - 15 秒自己紹介
  - 3 分技術説明
  - Q&A

**夜（2時間）**
- [ ] 十分な睡眠

#### Day 29（火）

**朝（1時間）**
- [ ] 朝食をしっかり取る
- [ ] リラックス

**昼（30分）**
- [ ] 面接前 30 分
  - マイク・カメラ確認
  - 緊張を取る深呼吸

**夕方（90 分）**
- [ ] **技術面接実施**
  - 自己紹介・プロジェクト説明
  - 技術Q&A
  - 逆質問
  - フィードバック受け取り

**夜（1時間）**
- [ ] 面接の感想をノート
- [ ] 改善点を整理

#### Day 30（水）

**朝（1時間）**
- [ ] 翌次ステップを確認
- [ ] 合否通知を待つ

**昼（1時間）**
- [ ] お礼メール送信（面接官へ）
  ```
  件名：本日はお忙しい中、面接いただきありがとうございました
  
  本文：
  本日は技術面接の機会をいただき、
  ありがとうございました。
  
  ご指摘いただいた点については、
  今後の改善に活かしていきたいと思います。
  
  ご質問ありがとうございました。
  ```

**夜（1時間）**
- [ ] 残り課題があれば改善
- [ ] 次のステップを待つ

---

## 📊 30日間の成果物チェックリスト

```
Week 1 - ポートフォリオ完成 ✅ 完了
✅ Colab で全 27 セル実行確認
✅ LoRA 学習完了（Loss=0.969）
✅ テスト全実行
✅ API 動作確認（v2.0.0）
✅ GitHub リポジトリ公開
✅ HuggingFace モデルアップロード（Shion1124/vlm-lora-agentic-rag）
☑️ GitHub Releases 作成

Week 2 - FastAPI 本番化・デプロイ ✅ 完了
✅ FastAPI v2.0.0（APIキー認証・マルチモーダル）
✅ Visual RAG (CLIP) 統合
✅ Agentic RAG 統合
✅ Docker コンテナ化（CPU/GPU 両対応）
✅ Cloud Run デプロイ成功
✅ 全エンドポイント動作確認
✅ 技術ブログ 5 本執筆完了

Week 3 - 面接対策
☑️ 技術理解を深掘り
☑️ 面接シミュレーション実施
☑️ 模擬面接（友人・メンター）
☑️ 逆質問 5+ 個準備
☑️ アーキテクチャ図を手書き化

Week 4 - 応募・面接
☑️ Stockmark に応募
☑️ 書類選考通過
☑️ 技術面接実施
☑️ 最終面接（オプション）
☑️ 内定獲得 🎉
```

---

## 🎁 最後のアドバイス

### 成功を確かにするために

```
1. 毎日のチェックリスト実行
   └─ 完璧を目指すのではなく「完了」を目指す

2. 質問・困ったときは聞く
   └─ メンター・友人に相談

3. 落ち込まない心構え
   └─ 面接はトライ＆ラーン
   └─ 不合格も学習機会

4. 自分を信じる
   └─ このポートフォリオの完成度は業界トップレベル
   └─ 自信を持ってプレゼンテーション

5. 面接官は敵ではなく同志
   └─ 相互理解の場
   └─ 自然に会話を
```

### 最終チェック（面接当日朝）

```
✓ 十分な睡眠（7時間以上）
✓ 朝食をしっかり食べた
✓ 身だしなみが整っている
✓ Zoom・Meet 接続テスト完了
✓ マイク・カメラ動作確認
✓ 背景が整理されている
✓ 照明が適切
✓ 資料が手元にある
✓ ノート・ペン準備
✓ 気持ちが前向き
```

---

## 🎯 成功の定義

```
【最高の成功】
✅ Stockmark から内定獲得
✅ 意気投合した開発チームと合意
✅ 今後のキャリアに納得

【良い成功】
✅ Stockmark の最終面接に進む
✅ フィードバックで成長機会を得る
✅ 次の機会へつながる

【学習の成功】
✅ VLM・RAG の理解が深まった
✅ 本番環境対応の知識を獲得
✅ ポートフォリオの形成経験
```

---

**You're ready. You've got this. Let's go! 🚀**

**30日後、あなたは Stockmark のチームの一員として**
**Visual RAG の最前線で活躍しているはずです。**

**Good luck! 💪**

