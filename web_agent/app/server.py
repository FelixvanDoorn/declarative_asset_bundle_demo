"""FastAPI server for the web agent: a JSON API under /api and the built React app at /."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("web_agent")

# Built by `npm run build` in frontend/ (see web_agent/README.md).
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    agent.init()  # Fails fast at startup with a clear message if settings are missing
    yield


app = FastAPI(title="Wiki agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(max_length=2000)


class Source(BaseModel):
    title: str
    chunk: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(request: AskRequest) -> AskResponse:
    # A plain `def` endpoint runs in FastAPI's thread pool, so questions are answered concurrently.
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please type a question.")
    try:
        return AskResponse(**agent.ask(question))
    except Exception as e:
        if agent.is_rate_limited(e):
            log.warning("Rate limited while answering: %s", e)
            raise HTTPException(
                status_code=429, detail="The model is busy right now (rate limit). Please try again in a moment."
            ) from e
        log.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail="Something went wrong while answering. Please try again.") from e


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
else:
    log.warning("Frontend not built (%s is missing); only the /api routes are available.", STATIC_DIR)


if __name__ == "__main__":
    # Databricks Apps sets DATABRICKS_APP_PORT; 8000 is used locally.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
