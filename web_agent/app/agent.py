"""The agent behind the web app: a LangChain/LangGraph agent that answers questions from Wikipedia.

Retrieval: the question is embedded with the same model the pipeline used (ai_query with
EMBEDDING_ENDPOINT), then the Delta Sync vector index (VS_INDEX_NAME) is queried with that
vector, because the index stores self-managed embeddings.

Nothing happens at import time: `init()` (called once at server startup) validates the
settings, configures MLflow tracing and builds the agent; `ask()` answers one question.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.errors import TooManyRequests
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from mlflow.entities import SpanType

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions using a Wikipedia search tool. Always call search_wiki first, "
    "answer only from the returned passages, and cite the article titles you used. "
    "If the tool finds no relevant passages, or they don't contain the answer, say you couldn't find it."
)
NOT_FOUND = "I couldn't find an answer to that in the indexed Wikipedia articles."


@dataclass(frozen=True)
class Settings:
    chat_endpoint: str
    embedding_endpoint: str
    index_name: str
    experiment_id: str | None
    num_results: int
    # Index scores are small for this index: good matches ~0.004-0.005, unrelated ~0.002.
    score_threshold: float

    @classmethod
    def from_env(cls) -> Settings:
        required = ("SERVING_ENDPOINT", "EMBEDDING_ENDPOINT", "VS_INDEX_NAME")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"Missing environment variables: {', '.join(missing)}. On Databricks Apps they come from "
                "resources/web_agent.app.yml; for local runs, export them (see web_agent/README.md)."
            )
        return cls(
            chat_endpoint=os.environ["SERVING_ENDPOINT"],
            embedding_endpoint=os.environ["EMBEDDING_ENDPOINT"],
            index_name=os.environ["VS_INDEX_NAME"],
            experiment_id=os.environ.get("MLFLOW_EXPERIMENT_ID") or None,
            num_results=int(os.environ.get("RETRIEVAL_NUM_RESULTS", "5")),
            score_threshold=float(os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.003")),
        )


@functools.cache
def settings() -> Settings:
    return Settings.from_env()


@functools.cache
def _workspace() -> WorkspaceClient:
    # Bounded HTTP timeouts and retries (the SDK retries 429/503 until retry_timeout_seconds).
    return WorkspaceClient(config=Config(http_timeout_seconds=30, retry_timeout_seconds=60))


def init() -> None:
    """Validate settings, configure tracing and build the agent. Call once at startup."""
    s = settings()
    mlflow.set_tracking_uri("databricks")
    if s.experiment_id:
        mlflow.set_experiment(experiment_id=s.experiment_id)
        mlflow.langchain.autolog()  # Traces the agent graph, LLM calls and tools
        log.info("Tracing to MLflow experiment %s", s.experiment_id)
    else:  # Typically a local run: don't mix local traces into another experiment
        mlflow.tracing.disable()
        log.info("MLflow tracing disabled (set MLFLOW_EXPERIMENT_ID to enable)")
    _agent()
    log.info("Agent ready: chat=%s embeddings=%s index=%s", s.chat_endpoint, s.embedding_endpoint, s.index_name)


def is_rate_limited(exc: BaseException) -> bool:
    """True if the error (or anything it was raised from) is an HTTP 429 / rate limit."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, TooManyRequests)
            or getattr(current, "status_code", None) == 429
            or type(current).__name__ == "RateLimitError"
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str) -> list[dict]:
    """Embed the query and return the closest wiki chunks above the score threshold."""
    s = settings()
    w = _workspace()
    vector = w.serving_endpoints.query(name=s.embedding_endpoint, input=[query]).data[0].embedding
    response = w.vector_search_indexes.query_index(
        index_name=s.index_name,
        columns=["title", "chunk"],
        query_vector=vector,
        num_results=s.num_results,
        score_threshold=s.score_threshold,
    )
    columns = [c.name for c in response.manifest.columns]  # title, chunk, score
    return [dict(zip(columns, row)) for row in (response.result.data_array or [])]


@tool(response_format="content_and_artifact")
def search_wiki(query: str) -> tuple[str, list[dict]]:
    """Search Wikipedia articles. Returns the most relevant passages with their article title."""
    passages = retrieve(query)
    if not passages:
        return "No relevant passages found.", []
    return "\n\n".join(f"[{p['title']}] {p['chunk']}" for p in passages), passages


@functools.cache
def _agent():
    llm = ChatDatabricks(endpoint=settings().chat_endpoint, temperature=0, timeout=60, max_retries=2)
    return create_agent(llm, [search_wiki], system_prompt=SYSTEM_PROMPT)


@mlflow.trace(name="web_agent")
def ask(question: str) -> dict:
    """Answer one question. Returns {"answer": str, "sources": [{"title", "chunk", "score"}]}."""
    messages = _agent().invoke({"messages": [{"role": "user", "content": question}]})["messages"]

    # The passages the agent actually retrieved, deduplicated, best score first.
    sources: dict[tuple[str, str], dict] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.artifact:
            for passage in message.artifact:
                sources.setdefault((passage["title"], passage["chunk"]), passage)
    ranked = sorted(sources.values(), key=lambda p: p["score"], reverse=True)

    # The model tends to fall back on its own knowledge when retrieval finds nothing, despite
    # the prompt; without sources the answer can't be checked, so don't show it.
    if not ranked:
        return {"answer": NOT_FOUND, "sources": []}
    return {"answer": messages[-1].text, "sources": ranked}
