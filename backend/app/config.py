import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    port: int = 3001
    frontend_origin: str = "http://localhost:5173"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    sqlite_path: str = "./db/astraml.db"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    model_cache_dir: str = "./model_cache"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        # If relative, resolve from workspace root / backend root
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / p
        return p

    @property
    def model_cache_path(self) -> Path:
        p = Path(self.model_cache_dir)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / p
        return p

    @property
    def sqlite_db_path(self) -> Path:
        p = Path(self.sqlite_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / p
        return p


settings = Settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("astraml")
