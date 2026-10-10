from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app.db.connection import init_db
from backend.app.db.repository import ChatRepository, ExperimentRepository
from backend.app.models.schemas import (
    AstraVerdict,
    ChatMessage,
    ModelResult,
    TrainingSession,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_astraml.db"
    init_db(db_file)
    return db_file


def create_sample_session(session_id: str = "sess-123") -> TrainingSession:
    return TrainingSession(
        session_id=session_id,
        dataset_id="dataset-abc",
        target_column="churn",
        problem_type="classification",
        best_model="random_forest",
        results=[
            ModelResult(
                algorithm="random_forest",
                accuracy=0.92,
                f1_score=0.91,
                precision=0.90,
                recall=0.92,
                training_time_ms=120,
            ),
            ModelResult(
                algorithm="logistic_regression",
                accuracy=0.85,
                f1_score=0.84,
                precision=0.83,
                recall=0.85,
                training_time_ms=50,
            ),
        ],
        astra_verdict=AstraVerdict(
            winner="random_forest",
            winner_reason="Random Forest won with F1 0.910.",
            key_insight="Feature signal was strongest in Random Forest.",
            recommendation="Deploy Random Forest model.",
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def test_init_db(temp_db: Path) -> None:
    assert temp_db.exists()


def test_experiment_repository_crud(temp_db: Path) -> None:
    repo = ExperimentRepository(db_path=temp_db)
    session = create_sample_session()

    # Create experiment
    exp = repo.create_experiment(session, dataset_name="customers.csv")
    assert exp.id is not None
    assert exp.dataset_filename == "customers.csv"
    assert exp.best_algorithm == "random_forest"
    assert exp.best_metric == 0.91
    assert exp.metric_name == "f1_score"

    # List experiments
    all_exps = repo.list_experiments()
    assert len(all_exps) == 1
    assert all_exps[0].id == exp.id

    # Get single experiment
    fetched = repo.get_experiment(exp.id)
    assert fetched is not None
    assert fetched.id == exp.id

    # Get training session by experiment ID
    fetched_session = repo.get_session_by_experiment(exp.id)
    assert fetched_session is not None
    assert fetched_session.session_id == session.session_id
    assert fetched_session.best_model == "random_forest"

    # Get training session by session ID
    by_session_id = repo.get_session_by_session_id(session.session_id)
    assert by_session_id is not None
    assert by_session_id.session_id == session.session_id

    # Delete experiment
    deleted = repo.delete_experiment(exp.id)
    assert deleted is True
    assert repo.get_experiment(exp.id) is None
    assert len(repo.list_experiments()) == 0


def test_chat_repository_crud(temp_db: Path) -> None:
    chat_repo = ChatRepository(db_path=temp_db)
    chat_id = "chat-xyz"

    # Get or create initial session
    data = chat_repo.get_or_create_chat_session(chat_id)
    assert data["id"] == chat_id
    assert data["messages"] == []

    # Update with messages
    msg1 = ChatMessage(
        id="msg-1",
        role="user",
        content="What is the top feature?",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    msg2 = ChatMessage(
        id="msg-2",
        role="assistant",
        content="The top feature is tenure.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    chat_repo.update_chat_session(chat_id, "exp-1", [msg1, msg2], {"model": "rf"})

    messages = chat_repo.get_messages(chat_id)
    assert len(messages) == 2
    assert messages[0].content == "What is the top feature?"
    assert messages[1].content == "The top feature is tenure."
