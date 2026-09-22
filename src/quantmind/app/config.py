"""Application Configuration Module (PRD v4.0 Productization)."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Centralized application configuration settings."""

    app_env: str = Field(default="development", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Authoritative core data directories and paths
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
    )
    core_db_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("QUANTMIND_CORE_DB_PATH", "data/quantmind_core.db")
        )
    )

    # Separate application-level metadata database (auth, jobs, user settings only)
    app_db_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("QUANTMIND_APP_DB_PATH", "data/quantmind_app.db")
        )
    )

    # Authentication & Security
    secret_key: str = Field(
        default="quantmind-insecure-dev-secret-key-change-in-production-2026",
        alias="AUTH_SECRET_KEY",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # CORS origins
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
    }


def get_settings() -> AppSettings:
    """Retrieve application settings instance."""
    return AppSettings()
