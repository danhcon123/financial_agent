from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3-vl:4b"
    OLLAMA_TEMPERATURE: float = 0.7


    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()