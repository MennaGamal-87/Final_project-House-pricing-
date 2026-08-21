from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "House Price Prediction API"
    api_version: str = "1.0.0"

    # Comma-separated list of allowed CORS origins
    cors_origins: str = "http://localhost:5173"

    # Paths to model artifacts (relative to backend/)
    model_path: str = "models/house_price.pkl"
    model_metadata_path: str = "models/model_metadata.json"
    locations_path: str = "models/locations.json"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_model_path(self) -> Path:
        return BASE_DIR / self.model_path

    @property
    def resolved_metadata_path(self) -> Path:
        return BASE_DIR / self.model_metadata_path

    @property
    def resolved_locations_path(self) -> Path:
        return BASE_DIR / self.locations_path


@lru_cache
def get_settings() -> Settings:
    return Settings()
