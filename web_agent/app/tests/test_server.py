import pytest
from databricks.sdk.errors import TooManyRequests
from fastapi.testclient import TestClient

import agent
import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(agent, "init", lambda: None)  # No Databricks/MLflow setup in tests
    with TestClient(server.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client, monkeypatch):
    seen = []

    def fake_ask(question):
        seen.append(question)
        return {"answer": "Northern Italy.", "sources": [{"title": "Ladin language", "chunk": "...", "score": 0.005}]}

    monkeypatch.setattr(agent, "ask", fake_ask)
    response = client.post("/api/ask", json={"question": "  Where is Ladin spoken?  "})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Northern Italy.",
        "sources": [{"title": "Ladin language", "chunk": "...", "score": 0.005}],
    }
    assert seen == ["Where is Ladin spoken?"]  # Whitespace is stripped


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(client, question):
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 422
    assert response.json() == {"detail": "Please type a question."}


def test_too_long_question_is_rejected(client):
    response = client.post("/api/ask", json={"question": "x" * 2001})
    assert response.status_code == 422


def test_rate_limit_returns_429_with_friendly_message(client, monkeypatch):
    def raise_rate_limit(question):
        raise TooManyRequests("Rate limit exceeded for endpoint databricks-gte-large-en")

    monkeypatch.setattr(agent, "ask", raise_rate_limit)
    response = client.post("/api/ask", json={"question": "hi"})
    assert response.status_code == 429
    assert "busy" in response.json()["detail"]


def test_unexpected_error_does_not_leak_details(client, monkeypatch):
    def raise_internal(question):
        raise RuntimeError("Index workspace.secret_schema.idx not found")

    monkeypatch.setattr(agent, "ask", raise_internal)
    response = client.post("/api/ask", json={"question": "hi"})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "secret_schema" not in detail
    assert detail == "Something went wrong while answering. Please try again."


@pytest.mark.skipif(not server.STATIC_DIR.is_dir(), reason="frontend not built (run npm run build in frontend/)")
def test_frontend_is_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<div id="root">' in response.text
