#!/usr/bin/env python3
"""
VLM + LoRA Agentic RAG - 実装検証スクリプト

このスクリプトは以下を検証します:
1. 必須ファイルの存在
2. Python依存関係の確認
3. api_production.py の構文チェック
4. LoRA 統合コードの確認
"""

import sys
import os
import json
from pathlib import Path

# ============================================================
# カラー出力
# ============================================================

class Colors:
    BOLD = '\033[1m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BOLD}{text}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}\n")

def print_step(text):
    print(f"{Colors.BOLD}{Colors.YELLOW}▶ {text}{Colors.NC}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.NC}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.NC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.NC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.NC}")

# ============================================================
# Step 1: ファイル検証
# ============================================================

def check_files():
    print_header("Step 1: 必須ファイル検証")
    
    required_files = {
        'api_production.py': 'FastAPI 本番実装',
        'vlm_agentic_rag_complete.py': 'VLM + LoRA パイプライン',
        'Dockerfile': 'Docker イメージ定義',
        'docker-compose.yml': 'Docker Compose 設定',
        'requirements_production.txt': '本番依存パッケージ',
        'vlm_agentic_rag_colab.ipynb': 'Colab 学習ノートブック',
    }
    
    all_exist = True
    for filename, description in required_files.items():
        print_step(f"{filename} ({description})")
        if Path(filename).exists():
            size = os.path.getsize(filename)
            print_success(f"存在（{size:,} bytes）")
        else:
            print_error(f"見つかりません")
            all_exist = False
    
    return all_exist

# ============================================================
# Step 2: Python コード構文チェック
# ============================================================

def check_python_syntax():
    print_header("Step 2: Python コード構文検証")
    
    python_files = [
        'api_production.py',
        'vlm_agentic_rag_complete.py',
    ]
    
    all_valid = True
    for filename in python_files:
        print_step(f"{filename} の構文チェック")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, filename, 'exec')
            print_success(f"構文OK")
        except SyntaxError as e:
            print_error(f"構文エラー: {e}")
            all_valid = False
    
    return all_valid

# ============================================================
# Step 3: LoRA 統合コード確認
# ============================================================

def check_lora_integration():
    print_header("Step 3: LoRA 統合コード確認")
    
    print_step("api_production.py での PeftModel インポート確認")
    with open('api_production.py', 'r') as f:
        content = f.read()
    
    checks = {
        'PeftModel インポート': 'from peft import PeftModel',
        'LoRA adapter ロード': 'PeftModel.from_pretrained',
        'HF リポジトリ参照': 'Shion1124/vlm-lora-agentic-rag',
        'lora_loaded フィールド': 'lora_loaded',
        'lora_adapter メタデータ': 'lora_adapter',
    }
    
    all_found = True
    for check_name, search_string in checks.items():
        if search_string in content:
            print_success(f"{check_name} ✓ 検出")
        else:
            print_error(f"{check_name} ✗ 未検出")
            all_found = False
    
    return all_found

# ============================================================
# Step 4: 依存パッケージ確認
# ============================================================

def check_dependencies():
    print_header("Step 4: 本番依存パッケージ確認")
    
    print_step("requirements_production.txt 確認")
    with open('requirements_production.txt', 'r') as f:
        requirements = f.read()
    
    essential_packages = {
        'fastapi': 'API フレームワーク',
        'uvicorn': 'ASGI サーバー',
        'torch': '深層学習フレームワーク',
        'transformers': 'トランスフォーマー',
        'peft': 'LoRA ライブラリ',  # 重要
        'bitsandbytes': '4-bit 量子化',
        'sentence-transformers': 'テキスト埋め込み',
        'faiss-cpu': 'ベクトル検索',
        'pdf2image': 'PDF 処理',
        'pillow': '画像処理',
    }
    
    all_found = True
    for package, description in essential_packages.items():
        if package.lower() in requirements.lower():
            # バージョン抽出
            for line in requirements.split('\n'):
                if package.lower() in line.lower():
                    print_success(f"{package:<20} ({description:<20}): {line.strip()}")
                    break
        else:
            print_error(f"{package:<20} ({description:<20}): 見つかりません")
            all_found = False
    
    return all_found

# ============================================================
# Step 5: Docker 設定確認
# ============================================================

def check_docker_config():
    print_header("Step 5: Docker 設定確認")
    
    print_step("Dockerfile 確認")
    with open('Dockerfile', 'r') as f:
        dockerfile_content = f.read()
    
    checks = {
        'NVIDIA CUDA ベースイメージ': 'nvidia/cuda',
        'LLaVA をクローン': 'git clone https://github.com/haotian-liu/LLaVA.git',
        'requirements インストール': 'pip install -r requirements_production.txt',
        'api_production.py コピー': 'COPY api_production.py',
        'uvicorn 起動': 'uvicorn api_production:app',
        'ポート 8000 公開': 'EXPOSE 8000',
        'Health Check': 'HEALTHCHECK',
    }
    
    all_found = True
    for check_name, search_string in checks.items():
        if search_string in dockerfile_content:
            print_success(f"{check_name} ✓")
        else:
            print_error(f"{check_name} ✗")
            all_found = False
    
    return all_found

# ============================================================
# Step 6: メタデータ確認
# ============================================================

def check_api_endpoints():
    print_header("Step 6: API エンドポイント確認")
    
    with open('api_production.py', 'r') as f:
        content = f.read()
    
    endpoints = {
        'GET /': 'ルートエンドポイント',
        'GET /health': 'ヘルスチェック',
        'POST /analyze': 'ドキュメント分析',
        'POST /search': 'セマンティック検索',
    }
    
    print_step("エンドポイント一覧")
    all_found = True
    for endpoint, description in endpoints.items():
        method, path = endpoint.split()
        search_string = f'@app.{method.lower()}("{path}")'
        if search_string in content or f"@app.{method.lower()}" in content and path in content:
            print_success(f"{endpoint:<20} - {description}")
        else:
            print_error(f"{endpoint:<20} - {description}")
            all_found = False
    
    return all_found

# ============================================================
# メイン実行
# ============================================================

def main():
    print(f"\n{Colors.BOLD}VLM + LoRA Agentic RAG - 実装検証${Colors.NC}\n")
    
    results = {
        'ファイル検証': check_files(),
        'Python 構文': check_python_syntax(),
        'LoRA 統合': check_lora_integration(),
        '依存パッケージ': check_dependencies(),
        'Docker 設定': check_docker_config(),
        'API エンドポイント': check_api_endpoints(),
    }
    
    # ============================================================
    # サマリー
    # ============================================================
    
    print_header("検証結果サマリー")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20} {status}")
    
    print(f"\n総合: {passed}/{total} パス")
    
    if passed == total:
        print_success("すべての検証に合格しました！")
        print_info("本番デプロイメント準備完了")
        print(f"\n{Colors.BOLD}次のステップ:{Colors.NC}")
        print("  1. Docker をインストール（未インストール）")
        print("     brew install --cask docker")
        print("")
        print("  2. Docker ビルド & テスト")
        print("     docker build -t vlm-agentic-rag:latest .")
        print("     docker run -p 8000:8000 vlm-agentic-rag:latest")
        print("")
        print("  3. GCP Cloud Run デプロイ")
        print("     gcloud init")
        print("     gcloud run deploy vlm-agentic-rag-api --source .")
        return 0
    else:
        print_error("いくつかの検証に失敗しました")
        print_warning("上記のエラーを修正してから再度実行してください")
        return 1

if __name__ == '__main__':
    sys.exit(main())
