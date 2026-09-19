"""
SatQuery AI — Application Settings
Pydantic v2 BaseSettings with .env file support.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceMode(str, Enum):
    """Inference execution mode."""
    MOCK = "MOCK"
    CUDA = "CUDA"
    CPU = "CPU"
    NEURAL = "NEURAL"


_backend_dir = Path(__file__).resolve().parent.parent
_default_data_dir = _backend_dir / "data"


class Settings(BaseSettings):
    """
    Central configuration for SatQuery AI backend.
    Values are read from environment variables or the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=str(_backend_dir / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Inference ────────────────────────────────────────────
    inference_mode: InferenceMode = InferenceMode.MOCK
    max_vram_gb: float = 12.0

    # ── LLM Synthesis Engine ─────────────────────────────────
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.8-flash"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # ── Paths ────────────────────────────────────────────────
    data_dir: Path = _default_data_dir
    checkpoints_dir: Path = _default_data_dir / "checkpoints"
    samples_dir: Path = _default_data_dir / "samples"
    upload_dir: Path = _default_data_dir / "uploads"
    report_dir: Path = _default_data_dir / "reports"

    # ── Upload Limits ────────────────────────────────────────
    max_upload_size_mb: int = 500

    # ── CORS ─────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,*"

    # ── Computed ─────────────────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> List[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",")]

    @field_validator("data_dir", "checkpoints_dir", "samples_dir", "upload_dir", "report_dir", mode="after")
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        resolved = v if v.is_absolute() else (_backend_dir / v).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


# Singleton instance
settings = Settings()
