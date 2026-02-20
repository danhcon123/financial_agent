from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    # Ollama config
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:8b"
    # ollama_model: str = "qwen3:30b"
    ollama_temperature: float = 0.0
    # System configuration
    log_level: str = "INFO"
    artifacts_root: str = "data/processed/runs"

@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()