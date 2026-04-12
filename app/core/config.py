from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Luôn đọc `.env` ở gốc package `data_ai_service/` (không phụ thuộc cwd khi chạy uvicorn từ repo root).
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

    # Internal Java Backend URL
    JAVA_BACKEND_URL: str = "http://localhost:8080/api/internal"
    INTERNAL_API_KEY: str = ""

    # AI (DeepSeek)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_TIMEOUT_SECONDS: int = 60


    # Local embedding model (OpenAI-compatible API)
    LOCAL_EMBEDDING_BASE_URL: str = ""
    LOCAL_EMBEDDING_API_KEY: str = "no-key-required"
    LOCAL_EMBEDDING_MODEL: str = ""

    # Transaction prefill behavior:
    # false => trust model output (strict mode), true => apply keyword-based category correction.
    PREFILL_ENABLE_CATEGORY_HEURISTIC: bool = False
    # Keep response structurally consistent for UI: if category type conflicts with tx type,
    # align tx type to the category type instead of leaving an inconsistent pair.
    PREFILL_ENFORCE_TYPE_CATEGORY_CONSISTENCY: bool = True

    # Vnstock
    VNSTOCK_API_KEY: str = ""

    # Export script / optional DB (same MySQL as Spring backend)
    FINFLOW_DATABASE_URL: str = ""
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = ""

    # Chat orchestration
    CHAT_TOOL_TIMEOUT_SECONDS: int = 30
    CHAT_RAG_ENABLED: bool = True
    CHAT_RAG_CHUNKS_DB: str = str(_SERVICE_ROOT / "artifacts" / "rag" / "annual_reports" / "chunks" / "annual_reports_chunks.sqlite")
    CHAT_QDRANT_URL: str = "http://127.0.0.1:6333"
    CHAT_QDRANT_API_KEY: str = ""
    CHAT_QDRANT_COLLECTION: str = "annual_report_chunks_bge_m3"
    CHAT_RAG_TOPK_VECTOR: int = 6
    CHAT_RAG_TOPK_KEYWORD: int = 6
    CHAT_RAG_TOPK_FINAL: int = 6
    CHAT_DEBUG_LOG_PROMPTS: bool = False
    CHAT_DEBUG_LOG_MAX_CHARS: int = 8000
    CHAT_FORECAST_ENABLED: bool = True
    CHAT_FORECAST_REPORT_TABLE_CSV: str = str(
        _SERVICE_ROOT / "artifacts" / "models" / "final_model_pipeline" / "report_table.csv"
    )
    CHAT_FORECAST_DETAIL_CSV: str = str(
        _SERVICE_ROOT / "artifacts" / "models" / "final_model_pipeline" / "predict_detail.csv"
    )
    CHAT_FORECAST_SUMMARY_JSON: str = str(
        _SERVICE_ROOT / "artifacts" / "models" / "final_model_pipeline" / "summary.json"
    )
    CHAT_FORECAST_ON_DEMAND_ENABLED: bool = True
    CHAT_FORECAST_ON_DEMAND_SCRIPT: str = str(
        _SERVICE_ROOT / "scripts" / "financial_training" / "test_final_models_forecast.py"
    )
    CHAT_FORECAST_ON_DEMAND_OUTPUT_DIR: str = str(
        _SERVICE_ROOT / "artifacts" / "models" / "final_model_pipeline" / "on_demand"
    )
    CHAT_FORECAST_ON_DEMAND_TIMEOUT_SECONDS: int = 180
    CHAT_FORECAST_TOP_FACTORS: int = 5

    # DeepSeek price snapshot (USD per 1M tokens)
    CHAT_DEEPSEEK_INPUT_PRICE_PER_1M: float = 0.10
    CHAT_DEEPSEEK_OUTPUT_PRICE_PER_1M: float = 0.40


settings = Settings()
