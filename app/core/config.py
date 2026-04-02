from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
