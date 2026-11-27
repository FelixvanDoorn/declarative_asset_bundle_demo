import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

mlflow.langchain.autolog()  # LangGraph uses LangChain's autolog
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/my-first-trace")


@tool
def get_weather(city: str):
    """Get weather for a city."""
    return f"It might be cloudy in {city}"


llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")  # Databricks Foundation Model API
agent = create_react_agent(llm, [get_weather])
result = agent.invoke({"messages": [("user", "What is the weather in SF?")]})
print(result["messages"][-1].content)
# Trace appears in the MLflow UI automatically
