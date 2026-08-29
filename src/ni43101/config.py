"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the extractor and critic providers."""

    extractor_api_key: SecretStr | None = None
    extractor_base_url: str = "https://api.deepseek.com"
    extractor_model: str = "deepseek-chat"

    critic_api_key: SecretStr | None = None
    critic_base_url: str | None = None
    critic_model: str = "glm-4.7-flash"

    max_revise_rounds: int = Field(default=3, ge=1, le=3)
    pass_score: float = Field(default=8, ge=8, le=10)
    field_tolerance: float = Field(default=0.05, ge=0.05, le=0.05)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
