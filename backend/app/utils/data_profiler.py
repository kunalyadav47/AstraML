from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from backend.app.config import logger, settings
from backend.app.models.schemas import (
    ClassBalance,
    ColumnProfile,
    DatasetProfile,
    TopCategory,
)
from backend.app.utils.leakage_detector import LeakageDetector
from backend.app.utils.paths import model_cache_dir

TARGET_CANDIDATES = [
    "target",
    "label",
    "y",
    "outcome",
    "class",
    "churn",
    "churned",
    "survived",
    "defaulted",
    "converted",
]


def clean_value(value: Any) -> str | int | float:
    if pd.isna(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        return value
    return str(value)


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_profile(dataset_id: str) -> DatasetProfile | None:
    profile_path = model_cache_dir() / "profiles" / f"{dataset_id}.json"
    if not profile_path.exists():
        return None
    return DatasetProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))


class DataProfiler:
    def detect_target(self, df: pd.DataFrame) -> str | None:
        lowered = {col.lower(): col for col in df.columns}
        for candidate in TARGET_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]

        for col in reversed(df.columns.tolist()):
            unique_count = df[col].nunique(dropna=True)
            if 1 < unique_count <= min(20, max(2, int(len(df) * 0.2))):
                return col
        return df.columns[-1] if len(df.columns) else None

    def infer_dtype(self, series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "unknown"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.85:
            return "datetime"
        if series.nunique(dropna=True) <= max(50, int(len(series) * 0.2)):
            return "categorical"
        return "unknown"

    def detect_problem_type(self, df: pd.DataFrame, target_col: str | None) -> str:
        if not target_col or target_col not in df.columns:
            return "unknown"
        target = df[target_col].dropna()
        if target.empty:
            return "unknown"
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"
        unique = target.nunique(dropna=True)
        if unique <= min(20, max(2, int(len(target) * 0.1))):
            return "classification"
        return "regression"

    def profile_column(self, name: str, series: pd.Series) -> ColumnProfile:
        dtype = self.infer_dtype(series)
        missing_count = int(series.isna().sum())
        sample_values = [clean_value(val) for val in series.dropna().head(5).tolist()]
        profile = ColumnProfile(
            name=name,
            dtype=dtype,  # type: ignore[arg-type]
            missing_count=missing_count,
            missing_pct=round(float(missing_count / len(series) * 100), 2) if len(series) else 0.0,
            unique_count=int(series.nunique(dropna=True)),
            sample_values=sample_values,
        )

        if dtype == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            profile.mean = round(float(numeric.mean()), 4) if numeric.notna().any() else None
            profile.std = round(float(numeric.std()), 4) if numeric.notna().sum() > 1 else None
            profile.min = round(float(numeric.min()), 4) if numeric.notna().any() else None
            profile.max = round(float(numeric.max()), 4) if numeric.notna().any() else None
        elif dtype in {"categorical", "boolean", "unknown"}:
            top = series.dropna().astype(str).value_counts().head(6)
            profile.top_categories = [
                TopCategory(value=str(val), count=int(count)) for val, count in top.items()
            ]
        return profile

    def build_summary(self, profile: DatasetProfile) -> str:
        missing_cols = [col.name for col in profile.columns if col.missing_count > 0]
        leakage = ", ".join(w.column for w in profile.leakage_warnings) or "none"
        balance = ""
        if profile.class_balance:
            total = sum(item.count for item in profile.class_balance)
            smallest = min(profile.class_balance, key=lambda item: item.count)
            balance = f" The smallest class is {smallest.label} at {smallest.count}/{total} rows."

        # Attempt Groq LLM summary if key provided
        if settings.groq_api_key and settings.groq_api_key != "your_groq_key_here":
            prompt = (
                "Summarize this dataset for a data scientist in 3 concise sentences. "
                "Use only the numbers provided.\n"
                f"Filename: {profile.filename}\n"
                f"Rows: {profile.row_count}, columns: {profile.col_count}\n"
                f"Target: {profile.target_column}, problem: {profile.problem_type}\n"
                f"Columns with missing values: {missing_cols or 'none'}\n"
                f"Leakage warnings: {leakage}\n"
                f"Class balance: {[item.model_dump() for item in profile.class_balance] if profile.class_balance else 'N/A'}"
            )
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "You are Astra, a precise data science copilot."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                    },
                    timeout=12,
                )
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    if content:
                        return content
            except Exception as exc:
                logger.debug("Groq dataset summary fallback: %s", exc)

        missing_text = (
            f"{len(missing_cols)} columns contain missing values"
            if missing_cols
            else "No missing values were detected"
        )
        target_text = (
            f"The likely target is {profile.target_column}, making this a {profile.problem_type} problem."
            if profile.target_column
            else "No clear target column was detected yet."
        )
        return (
            f"Astra read {profile.filename}: {profile.row_count} rows across {profile.col_count} columns. "
            f"{target_text} {missing_text}, and leakage checks flagged {leakage}.{balance}"
        )

    def profile(self, df: pd.DataFrame, dataset_id: str, filename: str) -> DatasetProfile:
        target_col = self.detect_target(df)
        problem_type = self.detect_problem_type(df, target_col)
        columns = [self.profile_column(name, df[name]) for name in df.columns]

        class_balance = None
        if target_col and problem_type == "classification":
            counts = df[target_col].dropna().astype(str).value_counts()
            class_balance = [
                ClassBalance(label=str(label), count=int(count)) for label, count in counts.items()
            ]

        numeric_df = df.select_dtypes(include="number")
        correlation_matrix: dict[str, dict[str, float]] = {}
        if 2 <= len(numeric_df.columns) <= 15:
            corr = numeric_df.corr(numeric_only=True).fillna(0)
            correlation_matrix = {
                row: {col: round(float(corr.loc[row, col]), 4) for col in corr.columns}
                for row in corr.index
            }

        leakage_warnings = LeakageDetector().check(df, target_col) if target_col else []
        profile = DatasetProfile(
            dataset_id=dataset_id,
            filename=filename,
            row_count=int(len(df)),
            col_count=int(len(df.columns)),
            columns=columns,
            target_column=target_col,
            problem_type=problem_type,  # type: ignore[arg-type]
            class_balance=class_balance,
            correlation_matrix=correlation_matrix,
            leakage_warnings=leakage_warnings,
            astra_summary="",
        )
        profile.astra_summary = self.build_summary(profile)

        profile_path = model_cache_dir() / "profiles" / f"{dataset_id}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        return profile
