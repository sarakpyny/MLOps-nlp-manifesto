"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_root_returns_message() -> None:
    """Root endpoint should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200

    payload = response.json()
    assert "message" in payload
    assert payload["docs"] == "/docs"


def test_health_returns_ok() -> None:
    """Health endpoint should confirm API readiness."""
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "analysis_experiment_dir" in payload
    assert "missing_local_files" in payload


def test_predict_topics_returns_valid_response() -> None:
    """Prediction endpoint should return expected fields."""
    response = client.post(
        "/predict_topics",
        json={
            "text": "Nous voulons défendre la justice sociale et les services publics."
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert "processed_text" in payload
    assert "top_topic_id" in payload
    assert "top_topic_label" in payload
    assert "top_topic_score" in payload
    assert "topic_distribution" in payload
    assert isinstance(payload["topic_distribution"], list)
    assert len(payload["topic_distribution"]) > 0


def test_predict_topics_rejects_empty_text() -> None:
    """Prediction endpoint should reject empty input."""
    response = client.post("/predict_topics", json={"text": ""})
    assert response.status_code in {400, 422}


def test_topics_returns_list() -> None:
    """Topics endpoint should return a non-empty topic list."""
    response = client.get("/topics")
    assert response.status_code == 200

    payload = response.json()
    assert "topics" in payload
    assert isinstance(payload["topics"], list)
    assert len(payload["topics"]) > 0


def test_stats_returns_summary() -> None:
    """Stats endpoint should return dataset summary fields."""
    response = client.get("/stats")
    assert response.status_code == 200

    payload = response.json()
    assert "n_documents" in payload
    assert "n_topics" in payload
    assert "dominant_topic_counts" in payload


def test_party_profile_returns_profile() -> None:
    """Party profile endpoint should return a profile for a known party."""
    response = client.get("/party_profile/Other")
    assert response.status_code == 200

    payload = response.json()
    assert payload["group_column"] == "party_clean"
    assert payload["group_value"] == "Other"
    assert "n_documents" in payload
    assert "top_topics" in payload


def test_profession_profile_returns_profile() -> None:
    """Profession profile endpoint should return a profile for a known profession."""
    response = client.get("/profession_profile/Health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["group_column"] == "profession_clean"
    assert payload["group_value"] == "Health"
    assert "n_documents" in payload
    assert "top_topics" in payload


def test_party_profile_returns_404_for_unknown_party() -> None:
    """Unknown party should return 404."""
    response = client.get("/party_profile/not_a_real_party")
    assert response.status_code == 404


def test_profession_profile_returns_404_for_unknown_profession() -> None:
    """Unknown profession should return 404."""
    response = client.get("/profession_profile/not_a_real_profession")
    assert response.status_code == 404
