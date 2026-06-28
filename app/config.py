"""Central configuration — single source of truth for paths and hyperparameters."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_DIR = DATA_DIR / "bm25"
LOGS_DIR = DATA_DIR / "logs"
EVAL_DATASET_PATH = ROOT / "app" / "evaluation" / "eval_dataset.json"

# Companies in the corpus (ticker → SEC CIK)
CORPUS_COMPANIES: dict[str, str] = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "META": "0001326801",
}

FILING_TYPES = ("10-K", "10-Q")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o-mini"

    sec_user_agent: str = "FinRAGEval contact@example.com"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    dense_weight: float = 0.5
    bm25_weight: float = 0.5

    faithfulness_alert_threshold: float = 0.7
    chunk_recall_alert_threshold: float = 0.5

    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_collection: str = "sec_filings"

    min_confidence_to_answer: float = 0.45
    min_faithfulness_to_answer: float = 0.55


settings = Settings()


def ensure_data_dirs() -> None:
    for d in (CORPUS_DIR, CHROMA_DIR, BM25_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)