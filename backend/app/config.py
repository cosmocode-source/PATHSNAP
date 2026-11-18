"""Configuration settings for PathSnap API backend.

Loads environment variables and .env file values using pydantic-settings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """PathSnap application settings."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Folder with the trips to serve (default ../Dataset/student_test)
    TRIPS_DIR: Path = BACKEND_DIR.parent / "Dataset" / "student_test"

    # Folder with the model files (default ../trained_models/model4)
    MODEL_DIR: Path = BACKEND_DIR.parent / "trained_models" / "model4"

    # Python path of the HMM decoder (format "module.path:function_name").
    # If blank or cannot be imported, API starts with hmm_available=false.
    HMM_MODULE: Optional[str] = None

    # If true, an HMM failure fails the trip; if false, returns classifier-only with warning.
    STRICT_HMM: bool = False

    # Server port
    PORT: int = 8000

    # Allowed CORS origins (comma-separated string or list of origins)
    ALLOWED_ORIGINS: Union[List[str], str] = ["*"]

    # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL: str = "INFO"

    # Thread pool size for CPU-heavy jobs
    MAX_WORKERS: int = 1

    # Time to retain completed jobs in memory (minutes)
    JOB_TTL_MINUTES: int = 60

    # Maximum runtime allowed per trip prediction (seconds)
    TRIP_TIMEOUT_S: int = 120

    # GPS downsampling threshold for map display
    MAX_DISPLAY_POINTS: int = 1000

    # Ngrok auth token (optional, from environment only)
    NGROK_AUTHTOKEN: Optional[str] = None

    @field_validator("TRIPS_DIR", "MODEL_DIR", mode="before")
    @classmethod
    def resolve_path(cls, v: Union[str, Path]) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = (BACKEND_DIR / p).resolve()
        return p

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


# Global singleton settings instance
settings = Settings()
