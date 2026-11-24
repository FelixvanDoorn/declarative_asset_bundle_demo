# web_agent

A minimal Gradio web UI (question box → answer) for the LangGraph agent, deployed as a
Databricks App. The agent runs inside the app and calls a Databricks Foundation Model
(`ChatDatabricks`); requests are traced to an MLflow experiment managed by this bundle.

```bash
cd web_agent
# Run locally (uses your Databricks CLI auth), then open http://localhost:8000
uv run --with-requirements app/requirements.txt app/app.py

databricks bundle deploy           # create/update the app + experiment
databricks bundle run web_agent    # deploy the code and start the app (prints the URL)
```

Free Edition: max 3 apps per account, and apps stop automatically 24 hours after a start or
redeploy — run `databricks bundle run web_agent` again to restart it.
