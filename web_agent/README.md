# web_agent

A question-answering web app over a sample of Wikipedia, deployed as a Databricks App:
a React (Vite + TypeScript) frontend served by a FastAPI backend that runs a LangChain agent.

```
wiki file (samples volume)
  → pipeline wiki_chunks: wiki_articles → wiki_chunks (chunk + ai_query embedding, CDF on)
  → Delta Sync vector index wiki_chunks_index on AI Search endpoint wiki-search
  → app: React UI → POST /api/ask (FastAPI) → embed question → query index → LLM answers
    from the passages, returned with its sources
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

# The built frontend (app/static/) is git-ignored: build it before every app deploy.
(cd app/frontend && npm ci && npm run build)
databricks bundle deploy && databricks bundle run web_agent   # upload and (re)start the app
```

### App layout

| Path | What |
| --- | --- |
| `app/server.py` | FastAPI: `POST /api/ask`, `GET /api/health`, serves `app/static/` at `/` |
| `app/agent.py` | Agent, retrieval (`search_wiki`), settings, tracing setup |
| `app/frontend/` | React + Vite + TypeScript source; `npm run build` writes `app/static/` |
| `app/requirements.in` → `requirements.txt` | Direct deps → fully pinned deps installed by Databricks Apps |

To change Python dependencies, edit `app/requirements.in` and regenerate:
`uv pip compile app/requirements.in -o app/requirements.txt --python-version 3.11 --python-platform linux`.

### Local development

```bash
# Terminal 1: API on :8000 (uses your Databricks CLI auth; tracing is off unless MLFLOW_EXPERIMENT_ID is set)
cd web_agent/app
export SERVING_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
export EMBEDDING_ENDPOINT=databricks-gte-large-en
export VS_INDEX_NAME=workspace.dev_<user>_web_agent.wiki_chunks_index
uv run --no-project --python 3.11 --with-requirements requirements.txt python server.py

# Terminal 2: frontend with hot reload on :5173 (proxies /api to :8000)
cd web_agent/app/frontend && npm install && npm run dev
```

### Tests

No Databricks access needed: the agent, workspace client and LLM are faked. CI runs the same
commands (`web-agent-checks` job in `.github/workflows/deploy.yml`).

```bash
cd web_agent/app/frontend && npm ci && npm run typecheck && npm test && npm run build
cd web_agent/app
uv run --no-project --python 3.11 --with-requirements requirements-dev.txt ruff check . --exclude frontend
uv run --no-project --python 3.11 --with-requirements requirements-dev.txt pytest -q
```

Optional settings: `RETRIEVAL_SCORE_THRESHOLD` (default 0.003; this index scores good matches
~0.004–0.005 and unrelated ones ~0.002) and `RETRIEVAL_NUM_RESULTS` (default 5). With no passage
above the threshold the app answers "couldn't find" instead of letting the model guess.

Variables (`databricks.yml`): `wiki_source_file` (default `part-00000.gz`), `wiki_sample_modulo`
(default 10, keeps ~10% of articles to limit embedding cost; 1 = all ~21k articles / ~77k chunks),
`embedding_endpoint` (default `databricks-gte-large-en`, 1024 dimensions; changing it requires
recreating the index).

Free Edition: one AI Search endpoint per workspace, max 3 apps, and apps stop 24 hours after a
start or redeploy (run `databricks bundle run web_agent` again).
