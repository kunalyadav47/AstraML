from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from backend.app.models.schemas import CleaningConfig, CleaningResult, PipelineStep
from backend.app.utils.data_profiler import DataProfiler, load_dataframe
from backend.app.utils.paths import upload_dir


class DataCleaner:
    def dataset_path(self, dataset_id: str) -> Path:
        directory = upload_dir()
        candidates = [
            directory / f"{dataset_id}.csv",
            directory / f"{dataset_id}.xlsx",
            directory / f"{dataset_id}.xls",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = list(directory.glob(f"{dataset_id}.*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Dataset {dataset_id} was not found.")

    def clean(self, dataset_id: str, config: CleaningConfig) -> CleaningResult:
        path = self.dataset_path(dataset_id)
        df = load_dataframe(path)
        original_shape = (int(df.shape[0]), int(df.shape[1]))
        rows_before = len(df)
        steps: list[PipelineStep] = []
        nulls_filled: dict[str, int] = {}
        columns_encoded: list[str] = []
        columns_scaled: list[str] = []

        target_col = DataProfiler().detect_target(df)

        # 1. Drop requested columns
        if config.columns_to_drop:
            valid = [col for col in config.columns_to_drop if col in df.columns]
            if valid:
                df = df.drop(columns=valid)
                steps.append(
                    PipelineStep(
                        step="Drop columns",
                        description=f"Dropped {len(valid)} selected columns: {', '.join(valid)}",
                        status="done",
                    )
                )
            else:
                steps.append(
                    PipelineStep(
                        step="Drop columns",
                        description="No selected columns existed in this dataset.",
                        status="skipped",
                    )
                )
        else:
            steps.append(
                PipelineStep(
                    step="Drop columns",
                    description="No manual column drops were requested.",
                    status="skipped",
                )
            )

        # 2. Handle missing values
        missing_before = int(df.isna().sum().sum())
        if missing_before:
            if config.missing_strategy == "drop_rows":
                before_len = len(df)
                df = df.dropna()
                steps.append(
                    PipelineStep(
                        step="Handle missing values",
                        description=f"Dropped {before_len - len(df)} rows containing missing values.",
                        status="done",
                    )
                )
            elif config.missing_strategy == "drop_cols":
                before_cols = df.shape[1]
                df = df.dropna(axis=1)
                steps.append(
                    PipelineStep(
                        step="Handle missing values",
                        description=f"Dropped {before_cols - df.shape[1]} columns containing missing values.",
                        status="done",
                    )
                )
            else:
                for col in df.columns:
                    missing = int(df[col].isna().sum())
                    if missing == 0:
                        continue
                    if pd.api.types.is_numeric_dtype(df[col]) and config.missing_strategy in {"mean", "median"}:
                        fill_value = (
                            float(df[col].mean())
                            if config.missing_strategy == "mean"
                            else float(df[col].median())
                        )
                    else:
                        mode = df[col].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else 0
                    df[col] = df[col].fillna(fill_value)
                    nulls_filled[col] = missing

                details = ", ".join(f"{count} in {col}" for col, count in nulls_filled.items())
                steps.append(
                    PipelineStep(
                        step="Handle missing values",
                        description=f"Filled {sum(nulls_filled.values())} null values using {config.missing_strategy}.",
                        status="done",
                        detail=details,
                    )
                )
        else:
            steps.append(
                PipelineStep(
                    step="Handle missing values",
                    description="No missing values were found.",
                    status="skipped",
                )
            )

        # 3. Handle outliers
        numeric_feature_cols = [
            col for col in df.select_dtypes(include="number").columns.tolist() if col != target_col
        ]
        if config.outlier_strategy == "none" or not numeric_feature_cols:
            steps.append(
                PipelineStep(
                    step="Handle outliers",
                    description="Outlier handling was skipped.",
                    status="skipped",
                )
            )
        elif config.outlier_strategy == "iqr_clip":
            clipped_cells = 0
            for col in numeric_feature_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or pd.isna(iqr):
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                before = df[col].copy()
                df[col] = df[col].clip(lower, upper)
                clipped_cells += int((before != df[col]).sum())

            steps.append(
                PipelineStep(
                    step="Handle outliers",
                    description=f"Clipped {clipped_cells} outlier values using IQR bounds.",
                    status="done" if clipped_cells else "skipped",
                )
            )
        else:  # zscore_remove
            before_len = len(df)
            mask = pd.Series(True, index=df.index)
            for col in numeric_feature_cols:
                std = df[col].std()
                if std == 0 or pd.isna(std):
                    continue
                z_scores = (df[col] - df[col].mean()).abs() / std
                mask &= z_scores <= 3
            df = df[mask]
            steps.append(
                PipelineStep(
                    step="Handle outliers",
                    description=f"Removed {before_len - len(df)} rows with z-score outliers.",
                    status="done" if before_len != len(df) else "skipped",
                )
            )

        # 4. Encode categorical features
        categorical_cols = [
            col
            for col in df.columns
            if col != target_col and not pd.api.types.is_numeric_dtype(df[col])
        ]
        if categorical_cols:
            if config.encoding_strategy == "onehot":
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
                columns_encoded = categorical_cols
            else:
                for col in categorical_cols:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                columns_encoded = categorical_cols
            steps.append(
                PipelineStep(
                    step="Encode categories",
                    description=f"Encoded {len(columns_encoded)} categorical columns using {config.encoding_strategy}.",
                    status="done",
                    detail=", ".join(columns_encoded),
                )
            )
        else:
            steps.append(
                PipelineStep(
                    step="Encode categories",
                    description="No categorical feature columns required encoding.",
                    status="skipped",
                )
            )

        # 5. Scale numeric features
        numeric_feature_cols = [
            col for col in df.select_dtypes(include="number").columns.tolist() if col != target_col
        ]
        if config.scaling_strategy == "none" or not numeric_feature_cols:
            steps.append(
                PipelineStep(
                    step="Scale numeric features",
                    description="Feature scaling was skipped.",
                    status="skipped",
                )
            )
        else:
            scaler = StandardScaler() if config.scaling_strategy == "standard" else MinMaxScaler()
            df[numeric_feature_cols] = scaler.fit_transform(
                df[numeric_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            )
            columns_scaled = numeric_feature_cols
            steps.append(
                PipelineStep(
                    step="Scale numeric features",
                    description=f"Scaled {len(columns_scaled)} numeric feature columns with {config.scaling_strategy}.",
                    status="done",
                )
            )

        # Save cleaned dataset
        cleaned_dataset_id = f"{dataset_id}_cleaned"
        cleaned_path = upload_dir() / f"{cleaned_dataset_id}.csv"
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cleaned_path, index=False)

        return CleaningResult(
            original_shape=original_shape,
            cleaned_shape=(int(df.shape[0]), int(df.shape[1])),
            rows_removed=int(rows_before - len(df)),
            columns_encoded=columns_encoded,
            columns_scaled=columns_scaled,
            nulls_filled=nulls_filled,
            cleaned_dataset_id=cleaned_dataset_id,
            pipeline_steps=steps,
        )
