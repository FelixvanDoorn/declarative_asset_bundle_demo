# script_agent

A minimal LangGraph agent using a Databricks Foundation Model (`ChatDatabricks`) with MLflow
tracing to the `/Shared/my-first-trace` experiment. It is its own bundle, separate from the ETL
bundle in the parent folder, so its dependencies are installed only for this job.

```bash
cd script_agent
uv run src/agent.py                        # run locally (uses your Databricks CLI auth)
databricks bundle deploy                   # deploy the job to the dev target
databricks bundle run script_agent         # one-off run on serverless compute
```
