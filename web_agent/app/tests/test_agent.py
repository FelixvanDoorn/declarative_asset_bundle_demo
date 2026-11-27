from types import SimpleNamespace

import pytest
from databricks.sdk.errors import TooManyRequests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent

# --- Settings ------------------------------------------------------------------------------


def test_settings_missing_env_lists_every_missing_variable(monkeypatch):
    for name in ("SERVING_ENDPOINT", "EMBEDDING_ENDPOINT", "VS_INDEX_NAME"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="SERVING_ENDPOINT, EMBEDDING_ENDPOINT, VS_INDEX_NAME"):
        agent.Settings.from_env()


def test_settings_from_env_with_defaults(monkeypatch):
    monkeypatch.setenv("SERVING_ENDPOINT", "chat")
    monkeypatch.setenv("EMBEDDING_ENDPOINT", "embed")
    monkeypatch.setenv("VS_INDEX_NAME", "c.s.i")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("RETRIEVAL_SCORE_THRESHOLD", raising=False)
    s = agent.Settings.from_env()
    assert (s.chat_endpoint, s.embedding_endpoint, s.index_name) == ("chat", "embed", "c.s.i")
    assert s.experiment_id is None
    assert s.num_results == 5
    assert s.score_threshold == 0.003


# --- Rate-limit detection --------------------------------------------------------------------


class RateLimitError(Exception):
    """Same class name as the OpenAI client's rate-limit error."""


class HTTPError(Exception):
    status_code = 429


def _raised_from(outer: RuntimeError, inner: Exception) -> RuntimeError:
    """Return `outer` as if raised from `inner` (sets __cause__ like `raise outer from inner`)."""
    try:
        raise outer from inner
    except RuntimeError as wrapped:
        return wrapped


@pytest.mark.parametrize(
    "error",
    [
        TooManyRequests("slow down"),
        RateLimitError("429"),
        HTTPError("too many"),
        _raised_from(RuntimeError("tool failed"), TooManyRequests("slow down")),
    ],
)
def test_is_rate_limited_true(error):
    assert agent.is_rate_limited(error)


def test_is_rate_limited_false_for_other_errors():
    assert not agent.is_rate_limited(ValueError("bad input"))
    assert not agent.is_rate_limited(_raised_from(RuntimeError("outer"), KeyError("inner")))


def test_is_rate_limited_handles_cyclic_context():
    a, b = ValueError("a"), ValueError("b")
    a.__context__, b.__context__ = b, a
    assert not agent.is_rate_limited(a)


# --- Retrieval -------------------------------------------------------------------------------


def test_retrieve_embeds_query_and_applies_settings(monkeypatch, test_settings):
    calls = {}

    def query_index(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            manifest=SimpleNamespace(columns=[SimpleNamespace(name=n) for n in ("title", "chunk", "score")]),
            result=SimpleNamespace(data_array=[["Coral reef", "Reefs are made of...", 0.0051]]),
        )

    fake_workspace = SimpleNamespace(
        serving_endpoints=SimpleNamespace(
            query=lambda name, input: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
        ),
        vector_search_indexes=SimpleNamespace(query_index=query_index),
    )
    monkeypatch.setattr(agent, "_workspace", lambda: fake_workspace)

    assert agent.retrieve("coral") == [{"title": "Coral reef", "chunk": "Reefs are made of...", "score": 0.0051}]
    assert calls["index_name"] == "cat.schema.index"
    assert calls["query_vector"] == [0.1, 0.2]
    assert calls["num_results"] == 5
    assert calls["score_threshold"] == 0.003


def test_retrieve_handles_no_results(monkeypatch, test_settings):
    fake_workspace = SimpleNamespace(
        serving_endpoints=SimpleNamespace(
            query=lambda name, input: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0])])
        ),
        vector_search_indexes=SimpleNamespace(
            query_index=lambda **_: SimpleNamespace(
                manifest=SimpleNamespace(columns=[SimpleNamespace(name="title")]),
                result=SimpleNamespace(data_array=None),
            )
        ),
    )
    monkeypatch.setattr(agent, "_workspace", lambda: fake_workspace)
    assert agent.retrieve("nothing") == []


def _tool_call(query: str) -> dict:
    return {"type": "tool_call", "id": "call-1", "name": "search_wiki", "args": {"query": query}}


def test_search_wiki_returns_text_for_the_llm_and_passages_as_artifact(monkeypatch):
    passages = [{"title": "Ladin language", "chunk": "Spoken in the Dolomites.", "score": 0.005}]
    monkeypatch.setattr(agent, "retrieve", lambda query: passages)
    message = agent.search_wiki.invoke(_tool_call("ladin"))
    assert message.content == "[Ladin language] Spoken in the Dolomites."
    assert message.artifact == passages


def test_search_wiki_without_passages(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda query: [])
    message = agent.search_wiki.invoke(_tool_call("sourdough"))
    assert message.content == "No relevant passages found."
    assert message.artifact == []


# --- ask() -----------------------------------------------------------------------------------


class FakeAgent:
    def __init__(self, messages):
        self.messages = messages
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return {"messages": self.messages}


def test_ask_returns_answer_with_deduplicated_sources_best_first(monkeypatch):
    low = {"title": "B", "chunk": "b", "score": 0.004}
    high = {"title": "A", "chunk": "a", "score": 0.006}
    fake = FakeAgent(
        [
            HumanMessage("q"),
            ToolMessage(content="...", artifact=[low, high], tool_call_id="1"),
            ToolMessage(content="...", artifact=[high], tool_call_id="2"),  # same passage again
            AIMessage("Final answer (A)"),
        ]
    )
    monkeypatch.setattr(agent, "_agent", lambda: fake)

    result = agent.ask("What is A?")

    assert fake.inputs == {"messages": [{"role": "user", "content": "What is A?"}]}
    assert result == {"answer": "Final answer (A)", "sources": [high, low]}


def test_ask_without_sources_returns_not_found_instead_of_model_guess(monkeypatch):
    fake = FakeAgent(
        [
            HumanMessage("q"),
            ToolMessage(content="No relevant passages found.", artifact=[], tool_call_id="1"),
            AIMessage("The capital is Ulaanbaatar, as generally known."),
        ]
    )
    monkeypatch.setattr(agent, "_agent", lambda: fake)
    assert agent.ask("Capital of Mongolia?") == {"answer": agent.NOT_FOUND, "sources": []}
