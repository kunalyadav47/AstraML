from __future__ import annotations

import pandas as pd
from backend.app.models.schemas import LeakageWarning


class LeakageDetector:
    LEAKAGE_THRESHOLD = 0.90
    WARNING_THRESHOLD = 0.70

    def check(self, df: pd.DataFrame, target_col: str) -> list[LeakageWarning]:
        warnings: list[LeakageWarning] = []
        if target_col not in df.columns or len(df) == 0:
            return warnings

        target = df[target_col]
        if not pd.api.types.is_numeric_dtype(target):
            target = pd.Series(pd.factorize(target.astype(str))[0], index=target.index)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        for col in numeric_cols:
            try:
                corr = abs(df[col].corr(target))
                if pd.isna(corr):
                    continue
                if corr >= self.LEAKAGE_THRESHOLD:
                    warnings.append(
                        LeakageWarning(
                            column=col,
                            correlation_with_target=round(float(corr), 3),
                            warning=(
                                f"'{col}' has {corr:.0%} correlation with target — "
                                "this is almost certainly a data leak. Remove before training."
                            ),
                        )
                    )
                elif corr >= self.WARNING_THRESHOLD:
                    warnings.append(
                        LeakageWarning(
                            column=col,
                            correlation_with_target=round(float(corr), 3),
                            warning=(
                                f"'{col}' has high correlation ({corr:.0%}) with target — "
                                "verify this is a legitimate feature, not future data."
                            ),
                        )
                    )
            except Exception:
                continue

        # Check for ID-like columns with near 1:1 unique values
        for col in df.select_dtypes(include="number").columns:
            if col == target_col or len(df) < 10:
                continue
            unique_ratio = df[col].nunique(dropna=True) / len(df)
            if unique_ratio > 0.95:
                warnings.append(
                    LeakageWarning(
                        column=col,
                        correlation_with_target=0.0,
                        warning=(
                            f"'{col}' appears to be an ID column ({df[col].nunique(dropna=True)} unique values). "
                            "ID columns should not be used as features."
                        ),
                    )
                )

        return warnings
