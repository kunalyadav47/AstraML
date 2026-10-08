from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class AstraBaseModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


ColumnDtype = Literal["numeric", "categorical", "datetime", "boolean", "unknown"]
ProblemType = Literal["classification", "regression", "unknown"]
TrainingProblemType = Literal["classification", "regression"]
Algorithm = Literal[
    "logistic_regression",
    "random_forest",
    "decision_tree",
    "knn",
    "gradient_boosting",
]
TrainingMode = Literal["fast", "balanced", "thorough"]
PipelineStatus = Literal["done", "skipped", "warning"]
ProgressType = Literal["start", "epoch", "model_done", "all_done", "error"]
ActionType = Literal[
    "retrain",
    "explain_feature",
    "compare_models",
    "clean_data",
    "suggest_next",
    "generate_report",
    "what_if",
    "detect_leakage",
    "show_confusion_matrix",
]


class TopCategory(AstraBaseModel):
    value: str
    count: int


class ColumnProfile(AstraBaseModel):
    name: str
    dtype: ColumnDtype
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[str | int | float]
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    top_categories: list[TopCategory] | None = None


class ClassBalance(AstraBaseModel):
    label: str
    count: int


class LeakageWarning(AstraBaseModel):
    column: str
    correlation_with_target: float
    warning: str


class DatasetProfile(AstraBaseModel):
    dataset_id: str
    filename: str
    row_count: int
    col_count: int
    columns: list[ColumnProfile]
    target_column: str | None = None
    problem_type: ProblemType = "unknown"
    class_balance: list[ClassBalance] | None = None
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    leakage_warnings: list[LeakageWarning] = Field(default_factory=list)
    astra_summary: str = ""


class CleaningConfig(AstraBaseModel):
    missing_strategy: Literal["mean", "median", "mode", "drop_rows", "drop_cols"] = "median"
    outlier_strategy: Literal["iqr_clip", "zscore_remove", "none"] = "iqr_clip"
    encoding_strategy: Literal["label", "onehot"] = "onehot"
    scaling_strategy: Literal["standard", "minmax", "none"] = "standard"
    columns_to_drop: list[str] = Field(default_factory=list)


class CleaningRequest(AstraBaseModel):
    dataset_id: str
    config: CleaningConfig


class PipelineStep(AstraBaseModel):
    step: str
    description: str
    status: PipelineStatus
    detail: str | None = None


class CleaningResult(AstraBaseModel):
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    rows_removed: int
    columns_encoded: list[str] = Field(default_factory=list)
    columns_scaled: list[str] = Field(default_factory=list)
    nulls_filled: dict[str, int] = Field(default_factory=dict)
    cleaned_dataset_id: str
    pipeline_steps: list[PipelineStep] = Field(default_factory=list)


class TrainingConfig(AstraBaseModel):
    dataset_id: str
    target_column: str
    problem_type: TrainingProblemType
    algorithms: list[Algorithm]
    mode: TrainingMode = "balanced"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    cross_validation_folds: int = Field(default=3, ge=3, le=5)


class TrainingProgressEvent(AstraBaseModel):
    type: ProgressType
    algorithm: Algorithm | None = None
    progress_pct: float | None = None
    current_metric: float | None = None
    message: str | None = None


class FeatureImportance(AstraBaseModel):
    feature: str
    importance: float
    shap_mean_abs: float | None = None


class ModelResult(AstraBaseModel):
    algorithm: Algorithm
    accuracy: float | None = None
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None
    roc_auc: float | None = None
    confusion_matrix: list[list[int]] | None = None
    rmse: float | None = None
    mae: float | None = None
    r2_score: float | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    training_time_ms: int = 0
    feature_importance: list[FeatureImportance] = Field(default_factory=list)


class AstraVerdict(AstraBaseModel):
    winner: Algorithm
    winner_reason: str
    key_insight: str
    recommendation: str
    watch_out: str | None = None


class TrainingSession(AstraBaseModel):
    session_id: str
    dataset_id: str
    target_column: str
    problem_type: TrainingProblemType
    results: list[ModelResult]
    best_model: Algorithm
    astra_verdict: AstraVerdict
    created_at: str


class ChatAction(AstraBaseModel):
    type: ActionType
    params: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result_summary: str | None = None


class ChatMessage(AstraBaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    action: ChatAction | None = None
    timestamp: str


class Experiment(AstraBaseModel):
    id: str
    name: str
    dataset_filename: str
    target_column: str
    problem_type: TrainingProblemType
    best_algorithm: Algorithm
    best_metric: float
    metric_name: str
    created_at: str
    session_id: str


class ReportRequest(AstraBaseModel):
    session_id: str
    format: Literal["pdf"] = "pdf"
    include_sections: list[str] = Field(default_factory=list)


class WhatIfRequest(AstraBaseModel):
    session_id: str
    feature_values: dict[str, Any] = Field(default_factory=dict)


class WhatIfResponse(AstraBaseModel):
    prediction: str | int | float
    confidence: float | None = None
    probabilities: list[dict[str, Any]] = Field(default_factory=list)
    contributions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ShapRequest(AstraBaseModel):
    session_id: str
    algorithm: Algorithm
    row_index: int | None = None
