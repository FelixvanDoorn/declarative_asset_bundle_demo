"""The agent behind the web UI: a LangGraph ReAct agent that answers questions from Wikipedia.

Retrieval: the question is embedded with the same model the pipeline used (ai_query with
EMBEDDING_ENDPOINT), then the Delta Sync vector index (VS_INDEX_NAME) is queried with that
vector, because the index stores self-managed embeddings.

On Databricks Apps all settings come from the app config in resources/web_agent.app.yml;
locally they come from environment variables (VS_INDEX_NAME is required).
"""

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from mlflow.entities import SpanType

CHAT_ENDPOINT = os.environ.get("SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
INDEX_NAME = os.environ["VS_INDEX_NAME"]  # e.g. workspace.dev_<user>_web_agent.wiki_chunks_index
NUM_RESULTS = 5

SYSTEM_PROMPT = (
    "You answer questions using a Wikipedia search tool. Always call search_wiki first, "
    "answer only from the returned passages, and cite the article titles you used. "
    "If the passages don't contain the answer, say you couldn't find it."
)

mlflow.set_tracking_uri("databricks")
if experiment_id := os.environ.get("MLFLOW_EXPERIMENT_ID"):
    mlflow.set_experiment(experiment_id=experiment_id)
else:  # Local runs
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/my-first-trace"))
mlflow.langchain.autolog()  # LangGraph uses LangChain's autolog

_w = WorkspaceClient()


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str) -> list[dict]:
    """Embed the query and return the closest wiki chunks from the vector index."""
    vector = _w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=[query]).data[0].embedding
    response = _w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        columns=["title", "chunk"],
        query_vector=vector,
        num_results=NUM_RESULTS,
    )
    columns = [c.name for c in response.manifest.columns]  # title, chunk, score
    return [dict(zip(columns, row)) for row in (response.result.data_array or [])]


@tool
def search_wiki(query: str) -> list[dict]:
    """Search Wikipedia articles. Returns the most relevant passages with their article title."""
    return retrieve(query)


_agent = create_react_agent(
    ChatDatabricks(endpoint=CHAT_ENDPOINT, temperature=0), [search_wiki], prompt=SYSTEM_PROMPT
)


@mlflow.trace(name="web_agent")
def ask(question: str) -> str:
    """Run one question through the agent and return the final answer (no streaming)."""
    result = _agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
