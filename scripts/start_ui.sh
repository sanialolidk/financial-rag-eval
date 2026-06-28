#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
exec streamlit run ui/streamlit_app.py --server.port 8501