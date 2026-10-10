from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.app.models.schemas import CleaningRequest, CleaningResult
from backend.app.services.cleaner import DataCleaner

router = APIRouter(prefix="/api/clean", tags=["Data Cleaning"])


@router.post("", response_model=CleaningResult)
async def clean_dataset(request: CleaningRequest) -> CleaningResult:
    try:
        return DataCleaner().clean(request.dataset_id, request.config)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cleaning failed: {exc}",
        ) from exc
