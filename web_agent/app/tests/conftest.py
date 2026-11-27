"""Tests run without Databricks: the agent, workspace client and LLM are replaced by fakes."""

import sys
from pathlib import Path

import mlflow
import pytest

# Make app modules (agent, server) importable when pytest runs from web_agent/app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent


@pytest.fixture(autouse=True)
def _no_tracing():
    # Keep tests from writing traces (or creating a local tracking store).
    mlflow.tracing.disable()
    yield


@pytest.fixture
def test_settings(monkeypatch):
    settings = agent.Settings(
        chat_endpoint="chat-model",
        embedding_endpoint="embedding-model",
        index_name="cat.schema.index",
        experiment_id=None,
        num_results=5,
        score_threshold=0.003,
    )
    monkeypatch.setattr(agent, "settings", lambda: settings)
    return settings
