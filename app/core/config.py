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
