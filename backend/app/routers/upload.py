from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.config import logger, settings
from backend.app.db.repository import ExperimentRepository
from backend.app.models.schemas import DatasetProfile
from backend.app.utils.data_profiler import DataProfiler, clean_value, load_dataframe
from backend.app.utils.paths import ensure_runtime_dirs, upload_dir

router = APIRouter(prefix="/api/upload", tags=["Dataset Upload"])
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@router.post("", response_model=DatasetProfile, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetProfile:
    ensure_runtime_dirs()

    original_name = file.filename or "dataset.csv"
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload a CSV or Excel file (.csv, .xlsx, .xls).",
        )

    dataset_id = str(uuid4())
    saved_path = upload_dir() / f"{dataset_id}{suffix}"

    # Read and stream file in chunks, validating max size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    total_bytes = 0

    try:
        with saved_path.open("wb") as output:
            while chunk := await file.read(1024 * 64):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    saved_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum size limit of {settings.max_file_size_mb} MB.",
                    )
                output.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        logger.error("Failed to write uploaded file %s: %s", original_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {exc}",
        ) from exc

    # Parse and validate dataframe
    try:
        df = load_dataframe(saved_path)
        if df.empty:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded dataset has no rows.",
            )
        if len(df.columns) < 2:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The dataset needs at least two columns.",
            )

        # For Excel files, persist a CSV copy as standard runtime format
        if suffix != ".csv":
            csv_path = upload_dir() / f"{dataset_id}.csv"
            df.to_csv(csv_path, index=False)

        # Profile the dataset and cache profile
        profile = DataProfiler().profile(df, dataset_id, original_name)
        ExperimentRepository().save_dataset_profile(profile)
        logger.info("Successfully analyzed dataset %s (%s rows, %s cols)", dataset_id, len(df), len(df.columns))
        return profile
    except HTTPException:
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        logger.error("Dataset analysis error on %s: %s", original_name, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not analyze dataset: {exc}",
        ) from exc


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 15) -> dict[str, Any]:
    directory = upload_dir()
    csv_path = directory / f"{dataset_id}.csv"
    if not csv_path.exists():
        matches = list(directory.glob(f"{dataset_id}.*"))
        if not matches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} was not found.",
            )
        csv_path = matches[0]

    try:
        df = load_dataframe(csv_path).head(limit)
        clean_rows = [
            {col: clean_value(row[col]) for col in df.columns}
            for _, row in df.iterrows()
        ]
        return {
            "dataset_id": dataset_id,
            "columns": df.columns.tolist(),
            "rows": clean_rows,
            "total_preview_rows": len(clean_rows),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read dataset preview: {exc}",
        ) from exc


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def get_dataset_profile(dataset_id: str) -> DatasetProfile:
    profile = ExperimentRepository().get_dataset_profile(dataset_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset profile for {dataset_id} was not found.",
        )
    return profile
