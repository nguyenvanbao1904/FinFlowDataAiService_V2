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

    # AI
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_TIMEOUT_SECONDS: int = 60

    # Hybrid routing defaults:
    # - local for simple structured tasks (prefill, typo-fix, chunk normalization)
    # - google for complex analysis (stock reasoning, deep recommendations)
    SIMPLE_LLM_PROVIDER: str = "local"
    COMPLEX_LLM_PROVIDER: str = "google"

    # Fast local model profile for simple tasks.
    LOCAL_LLM_BASE_URL: str = "http://127.0.0.1:9090/v1"
    LOCAL_LLM_API_KEY: str = "no-key-required"
    LOCAL_LLM_MODEL: str = "mlx-community/gemma-4-e2b-it-4bit"
    # on: always think, off: never think, auto: think first and fallback to no-think retry.
    LOCAL_LLM_THINKING_MODE: str = "auto"

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


settings = Settings()
