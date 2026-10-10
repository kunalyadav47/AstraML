from pathlib import Path
from backend.app.config import settings


def ensure_runtime_dirs() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.model_cache_path.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    (settings.model_cache_path / "profiles").mkdir(parents=True, exist_ok=True)
    (settings.model_cache_path / "sessions").mkdir(parents=True, exist_ok=True)


def upload_dir() -> Path:
    ensure_runtime_dirs()
    return settings.upload_path


def model_cache_dir() -> Path:
    ensure_runtime_dirs()
    return settings.model_cache_path
