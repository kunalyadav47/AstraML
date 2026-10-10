import io
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_clean_dataset_end_to_end() -> None:
    # 1. Upload sample dataset with missing values, categorical feature, and outliers
    csv_content = (
        "age,income,category,irrelevant_id,target\n"
        "25,50000,premium,id_1,0\n"
        "30,,basic,id_2,0\n"
        "45,120000,standard,id_3,1\n"
        ",35000,basic,id_4,1\n"
        "35,500000,premium,id_5,0\n"
        "40,85000,standard,id_6,1\n"
    )
    upload_res = client.post(
        "/api/upload",
        files={"file": ("dataset_for_cleaning.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    # 2. Request clean
    clean_payload = {
        "dataset_id": dataset_id,
        "config": {
            "columns_to_drop": ["irrelevant_id"],
            "missing_strategy": "median",
            "outlier_strategy": "iqr_clip",
            "encoding_strategy": "onehot",
            "scaling_strategy": "standard",
        },
    }
    clean_res = client.post("/api/clean", json=clean_payload)
    assert clean_res.status_code == 200
    data = clean_res.json()

    assert data["cleaned_dataset_id"] == f"{dataset_id}_cleaned"
    assert "income" in data["nulls_filled"] or "age" in data["nulls_filled"]
    assert "category" in data["columns_encoded"]
    assert len(data["pipeline_steps"]) == 5
    assert all(step["status"] in {"done", "skipped"} for step in data["pipeline_steps"])


def test_clean_missing_drop_rows() -> None:
    csv_content = (
        "x,y,target\n"
        "1,10,0\n"
        ",20,1\n"
        "3,30,0\n"
    )
    upload_res = client.post(
        "/api/upload",
        files={"file": ("drop_rows.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    clean_res = client.post(
        "/api/clean",
        json={
            "dataset_id": dataset_id,
            "config": {
                "columns_to_drop": [],
                "missing_strategy": "drop_rows",
                "outlier_strategy": "none",
                "encoding_strategy": "label",
                "scaling_strategy": "none",
            },
        },
    )
    assert clean_res.status_code == 200
    data = clean_res.json()
    assert data["rows_removed"] == 1
    assert data["cleaned_shape"][0] == 2


def test_clean_dataset_not_found() -> None:
    res = client.post(
        "/api/clean",
        json={
            "dataset_id": "non-existent-uuid-1234",
            "config": {
                "columns_to_drop": [],
                "missing_strategy": "mean",
                "outlier_strategy": "none",
                "encoding_strategy": "label",
                "scaling_strategy": "none",
            },
        },
    )
    assert res.status_code == 404
