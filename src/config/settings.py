from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False
    )
    ollama_mode: str = "local"

    # Local Ollama config
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:8b"

    # Ollama Cloud
    ollama_cloud_base_url: str = "https://ollama.com"
    ollama_cloud_model: str = "qwen3.5:397b-cloud"
    ollama_api_key: Optional[str] = None

    # ollama_model: str = "qwen3:30b"
    ollama_temperature: float = 0.0
    # System configuration
    log_level: str = "INFO"
    artifacts_root: str = "data/processed/runs"

    # Alpha Vantage API
    alpha_vantage_api_key: Optional[str] = None

    # GDELT (no key needed)
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2"

    
@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings() 