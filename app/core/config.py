from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Always read `.env` from data_ai_service/ root, regardless of cwd.
_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _SERVICE_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "FinFlow Data & AI Service"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "production"

    # Internal Java Backend
    JAVA_BACKEND_URL: str = "http://localhost:8080/api/internal"
    INTERNAL_API_KEY: str = ""

    # DeepSeek LLM
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    LLM_TIMEOUT_SECONDS: int = 60
    CHAT_DEEPSEEK_INPUT_PRICE_PER_1M: float = 0.10
    CHAT_DEEPSEEK_OUTPUT_PRICE_PER_1M: float = 0.40

    # Voyage embedding for RAG vector search.
    VOYAGE_EMBED_BASE_URL: str = "https://api.voyageai.com/v1"
    VOYAGE_EMBED_MODEL: str = "voyage-3.5-lite"
    VOYAGE_EMBED_INPUT_TYPE: str = "query"

    # FireAnt (financial data, company profile, ICB tree)
    FIREANT_ACCESS_TOKEN: str = ""
    FIREANT_API_BASE: str = "https://restv2.fireant.vn"
    CRAWLER_STATE_DIR: str = str(_SERVICE_ROOT / "artifacts" / "crawler")

    # MySQL (used by export scripts / batch crawler — not by the FastAPI runtime)
    FINFLOW_DATABASE_URL: str = ""
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = ""

    # Chat orchestration
    CHAT_TOOL_TIMEOUT_SECONDS: int = 30
    CHAT_RAG_ENABLED: bool = True
    CHAT_RAG_CHUNKS_DB: str = str(
        _SERVICE_ROOT / "artifacts" / "rag" / "annual_reports" / "chunks"
        / "annual_reports_chunks.sqlite"
    )
    CHAT_QDRANT_URL: str = "http://127.0.0.1:6333"
    CHAT_QDRANT_API_KEY: str = ""
    CHAT_QDRANT_COLLECTION: str = "annual_report_chunks_voyage_3_5_lite"
    CHAT_RAG_TOPK_VECTOR: int = 50
    CHAT_RAG_TOPK_KEYWORD: int = 50
    CHAT_RAG_TOPK_FINAL: int = 5
    CHAT_RAG_CONTEXT_MAX_CHARS: int = 1800

    # Voyage reranker re-scores hybrid-retrieved chunks after vector + keyword search.
    CHAT_RAG_RERANK_ENABLED: bool = True
    VOYAGE_RERANK_URL: str = "https://api.voyageai.com/v1/rerank"
    VOYAGE_RERANK_MODEL: str = "rerank-2.5-lite"
    VOYAGE_API_KEY: str = ""
    # Number of candidates fed into the reranker before keeping TOPK_FINAL.
    CHAT_RAG_RERANK_CANDIDATES: int = 64
    CHAT_RAG_RERANK_TIMEOUT_SECONDS: int = 30
    CHAT_DEBUG_LOG_PROMPTS: bool = False
    CHAT_DEBUG_LOG_MAX_CHARS: int = 8000

    # Per-request trace files: dump full LLM I/O for every chat orchestrate
    # request to a JSON file. Use for debugging — leave OFF in production
    # unless tracking a specific issue (each turn ≈ 10-50 KB).
    CHAT_TRACE_ENABLED: bool = False
    CHAT_TRACE_DIR: str = str(_SERVICE_ROOT / "artifacts" / "chat_traces")

    # Forecast (ML model artifacts produced by scripts/financial_training)
    CHAT_FORECAST_ENABLED: bool = True
    CHAT_FORECAST_SUMMARY_JSON: str = str(
        _SERVICE_ROOT / "artifacts" / "models" / "production_pipeline" / "summary.json"
    )
    CHAT_FORECAST_SCRIPT: str = str(
        _SERVICE_ROOT / "scripts" / "financial_training" / "test_final_models_forecast.py"
    )
    CHAT_FORECAST_TIMEOUT_SECONDS: int = 180
    CHAT_FORECAST_TOP_FACTORS: int = 5


settings = Settings()
