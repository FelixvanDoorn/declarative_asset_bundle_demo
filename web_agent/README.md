# web_agent

A minimal Gradio web UI (question box → answer) for a LangGraph agent that answers questions
from Wikipedia, deployed as a Databricks App.

```
wiki file (samples volume)
  → pipeline wiki_chunks: wiki_articles → wiki_chunks (chunk + ai_query embedding, CDF on)
  → Delta Sync vector index wiki_chunks_index on AI Search endpoint wiki-search
  → app: embed question → query index → LLM answers from the passages
```

| Piece | Where |
| --- | --- |
| Schema (`web_agent`, `dev_<user>_web_agent` in dev) + grant for the app | `resources/wiki.schema.yml` |
| Chunking + embedding pipeline (SQL) | `resources/wiki_chunks.pipeline.yml`, `src/pipeline/` |
| AI Search endpoint | `resources/wiki_search.endpoint.yml` |
| Job: refresh pipeline, then create/sync index | `resources/wiki_index.job.yml`, `src/jobs/sync_index.py` |
| App, its config and permissions | `resources/web_agent.app.yml`, `app/` |
| Trace experiment | `resources/web_agent_traces.experiment.yml` |

```bash
cd web_agent
databricks bundle deploy           # app, schema, pipeline, endpoint, job, experiment
databricks bundle run wiki_index   # chunks + embeddings, then create/sync the vector index
databricks bundle run web_agent    # deploy the app code and start the app (prints the URL)

# Run the app locally (uses your Databricks CLI auth), then open http://localhost:8000
VS_INDEX_NAME=workspace.dev_<user>_web_agent.wiki_chunks_index \
  uv run --with-requirements app/requirements.txt app/app.py
```

Variables (`databricks.yml`): `wiki_source_file` (default `part-00000.gz`), `wiki_sample_modulo`
(default 10, keeps ~10% of articles to limit embedding cost; 1 = all ~21k articles / ~77k chunks),
`embedding_endpoint` (default `databricks-gte-large-en`, 1024 dimensions; changing it requires
recreating the index).

Free Edition: one AI Search endpoint per workspace, max 3 apps, and apps stop 24 hours after a
start or redeploy (run `databricks bundle run web_agent` again).
