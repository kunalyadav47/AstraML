from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.config import settings
from backend.app.db.connection import get_db
from backend.app.models.schemas import (
    ChatMessage,
    DatasetProfile,
    Experiment,
    ModelResult,
    TrainingSession,
)


def display_algorithm(algorithm: str) -> str:
    return algorithm.replace("_", " ").title()


def metric_for_session(session: TrainingSession) -> tuple[str, float]:
    best = next(
        (r for r in session.results if r.algorithm == session.best_model),
        session.results[0] if session.results else None,
    )
    if not best:
        return ("f1_score" if session.problem_type == "classification" else "r2_score", 0.0)

    if session.problem_type == "classification":
        metric_name = "f1_score"
        val = best.f1_score if best.f1_score is not None else (best.accuracy or best.cv_mean or 0.0)
    else:
        metric_name = "r2_score"
        val = best.r2_score if best.r2_score is not None else (best.cv_mean or 0.0)
    return (metric_name, round(float(val), 4))


class ExperimentRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path

    @property
    def profiles_dir(self) -> Path:
        p = settings.model_cache_path / "profiles"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        file_path = self.profiles_dir / f"{profile.dataset_id}.json"
        file_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        clean_id = dataset_id.removesuffix("_cleaned")
        file_path = self.profiles_dir / f"{clean_id}.json"
        if not file_path.exists():
            return None
        return DatasetProfile.model_validate_json(file_path.read_text(encoding="utf-8"))

    def create_experiment(
        self,
        session: TrainingSession,
        dataset_name: str | None = None,
    ) -> Experiment:
        profile = self.get_dataset_profile(session.dataset_id)
        metric_name, best_metric = metric_for_session(session)
        exp_id = str(uuid4())

        resolved_dataset_name = dataset_name or (profile.filename if profile else session.dataset_id)
        now_dt = datetime.now(timezone.utc)
        name = f"{resolved_dataset_name} -> {display_algorithm(session.best_model)} ({now_dt.strftime('%b %d, %Y')})"

        experiment = Experiment(
            id=exp_id,
            name=name,
            dataset_filename=resolved_dataset_name,
            target_column=session.target_column,
            problem_type=session.problem_type,
            best_algorithm=session.best_model,
            best_metric=best_metric,
            metric_name=metric_name,
            created_at=session.created_at,
            session_id=session.session_id,
        )

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, name, dataset_filename, target_column, problem_type,
                    best_algorithm, best_metric, metric_name, session_blob, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.dataset_filename,
                    experiment.target_column,
                    experiment.problem_type,
                    experiment.best_algorithm,
                    experiment.best_metric,
                    experiment.metric_name,
                    session.model_dump_json(),
                    experiment.created_at,
                ),
            )
        return experiment

    def list_experiments(self) -> list[Experiment]:
        with get_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, name, dataset_filename, target_column, problem_type,
                       best_algorithm, best_metric, metric_name, created_at,
                       json_extract(session_blob, '$.session_id') AS session_id
                FROM experiments
                ORDER BY created_at DESC
                """
            ).fetchall()

        experiments: list[Experiment] = []
        for row in rows:
            experiments.append(
                Experiment(
                    id=row["id"],
                    name=row["name"],
                    dataset_filename=row["dataset_filename"],
                    target_column=row["target_column"],
                    problem_type=row["problem_type"],
                    best_algorithm=row["best_algorithm"],
                    best_metric=row["best_metric"],
                    metric_name=row["metric_name"],
                    created_at=row["created_at"],
                    session_id=row["session_id"] or "",
                )
            )
        return experiments

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with get_db(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, name, dataset_filename, target_column, problem_type,
                       best_algorithm, best_metric, metric_name, created_at,
                       json_extract(session_blob, '$.session_id') AS session_id
                FROM experiments
                WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()

        if not row:
            return None
        return Experiment(
            id=row["id"],
            name=row["name"],
            dataset_filename=row["dataset_filename"],
            target_column=row["target_column"],
            problem_type=row["problem_type"],
            best_algorithm=row["best_algorithm"],
            best_metric=row["best_metric"],
            metric_name=row["metric_name"],
            created_at=row["created_at"],
            session_id=row["session_id"] or "",
        )

    def get_session_by_experiment(self, experiment_id: str) -> TrainingSession | None:
        with get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT session_blob FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()

        if not row:
            return None
        return TrainingSession.model_validate_json(row["session_blob"])

    def get_session_by_session_id(self, session_id: str) -> TrainingSession | None:
        with get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT session_blob FROM experiments WHERE json_extract(session_blob, '$.session_id') = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return None
        return TrainingSession.model_validate_json(row["session_blob"])

    def delete_experiment(self, experiment_id: str) -> bool:
        with get_db(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            return cursor.rowcount > 0


class ChatRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path

    def get_or_create_chat_session(
        self,
        chat_session_id: str,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        with get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (chat_session_id,),
            ).fetchone()

            if row:
                return {
                    "id": row["id"],
                    "experiment_id": row["experiment_id"],
                    "messages": [
                        ChatMessage.model_validate(m)
                        for m in json.loads(row["messages_blob"])
                    ],
                    "context": json.loads(row["context_blob"]),
                }

            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO chat_sessions (id, experiment_id, messages_blob, context_blob, created_at, updated_at)
                VALUES (?, ?, '[]', '{}', ?, ?)
                """,
                (chat_session_id, experiment_id, now_iso, now_iso),
            )
            return {
                "id": chat_session_id,
                "experiment_id": experiment_id,
                "messages": [],
                "context": {},
            }

    def update_chat_session(
        self,
        chat_session_id: str,
        experiment_id: str | None,
        messages: list[ChatMessage],
        context: dict[str, Any] | None = None,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        messages_blob = json.dumps([m.model_dump() for m in messages])
        context_blob = json.dumps(context or {})

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET experiment_id = COALESCE(?, experiment_id),
                    messages_blob = ?,
                    context_blob = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (experiment_id, messages_blob, context_blob, now_iso, chat_session_id),
            )

    def get_messages(self, chat_session_id: str) -> list[ChatMessage]:
        session_data = self.get_or_create_chat_session(chat_session_id)
        return session_data["messages"]
