"""The agent behind the web UI: a LangGraph ReAct agent on a Databricks Foundation Model.

Same agent as script_agent/, but built once at import and exposed as `ask(question)`.
On Databricks Apps, SERVING_ENDPOINT and MLFLOW_EXPERIMENT_ID come from the app's
resources (see app.yaml); locally they fall back to the defaults below.
"""

import os

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

ENDPOINT = os.environ.get("SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

mlflow.set_tracking_uri("databricks")
if experiment_id := os.environ.get("MLFLOW_EXPERIMENT_ID"):
    mlflow.set_experiment(experiment_id=experiment_id)
else:  # Local runs
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/my-first-trace"))
mlflow.langchain.autolog()  # LangGraph uses LangChain's autolog


@tool
def get_weather(city: str):
    """Get weather for a city."""
    return f"It might be cloudy in {city}"


_agent = create_react_agent(ChatDatabricks(endpoint=ENDPOINT, temperature=0), [get_weather])


@mlflow.trace(name="web_agent")
def ask(question: str) -> str:
    """Run one question through the agent and return the final answer (no streaming)."""
    result = _agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
