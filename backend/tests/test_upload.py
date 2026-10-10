import io
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_upload_csv_success() -> None:
    csv_content = (
        "age,income,credit_score,churn\n"
        "25,50000,700,0\n"
        "30,65000,720,0\n"
        "45,120000,800,1\n"
        "22,35000,650,1\n"
        "35,80000,710,0\n"
    )
    file_bytes = io.BytesIO(csv_content.encode("utf-8"))

    response = client.post(
        "/api/upload",
        files={"file": ("customers.csv", file_bytes, "text/csv")},
    )

    assert response.status_code == 201
    data = response.json()
    assert "dataset_id" in data
    assert data["filename"] == "customers.csv"
    assert data["row_count"] == 5
    assert data["col_count"] == 4
    assert data["target_column"] == "churn"
    assert data["problem_type"] == "classification"
    assert len(data["columns"]) == 4

    dataset_id = data["dataset_id"]

    # Test preview
    preview_res = client.get(f"/api/upload/{dataset_id}/preview")
    assert preview_res.status_code == 200
    preview_data = preview_res.json()
    assert preview_data["total_preview_rows"] == 5
    assert len(preview_data["columns"]) == 4

    # Test profile retrieval
    profile_res = client.get(f"/api/upload/{dataset_id}/profile")
    assert profile_res.status_code == 200
    assert profile_res.json()["dataset_id"] == dataset_id


def test_upload_invalid_extension() -> None:
    bad_bytes = io.BytesIO(b"binary content")
    response = client.post(
        "/api/upload",
        files={"file": ("image.png", bad_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "Upload a CSV or Excel file" in response.json()["detail"]


def test_upload_empty_dataset() -> None:
    empty_bytes = io.BytesIO(b"")
    response = client.post(
        "/api/upload",
        files={"file": ("empty.csv", empty_bytes, "text/csv")},
    )
    assert response.status_code == 400
