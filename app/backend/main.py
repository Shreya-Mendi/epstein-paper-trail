"""
main.py — FastAPI backend for Paper Trail.

Endpoints:
  GET  /people          — all labeled individuals with consequence tiers
  POST /chat            — RAG chatbot query
  GET  /timeline        — timeline events sorted by date
  GET  /person/{name}   — profile card for a specific individual
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Load .env from repo root if present
_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classifier import get_all_labeled_persons, predict_for_person
from ner import extract_persons
from rag import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static timeline events (can be extended from corpus later)
# ---------------------------------------------------------------------------

TIMELINE_EVENTS = [
    {
        "date": "1994-01-01",
        "person": "Jeffrey Epstein",
        "event": "Epstein begins teaching at the Dalton School in New York.",
        "tier": 0,
    },
    {
        "date": "1998-01-01",
        "person": "Jeffrey Epstein",
        "event": "Epstein purchases Little St. James island in the U.S. Virgin Islands.",
        "tier": 0,
    },
    {
        "date": "2005-03-01",
        "person": "Jeffrey Epstein",
        "event": "Palm Beach police begin investigation after complaint filed.",
        "tier": 0,
    },
    {
        "date": "2007-09-24",
        "person": "Alexander Acosta",
        "event": "Non-prosecution agreement (NPA) secretly signed by Acosta and Epstein's lawyers.",
        "tier": 2,
    },
    {
        "date": "2008-06-30",
        "person": "Jeffrey Epstein",
        "event": "Epstein pleads guilty to Florida state charges; begins 18-month sentence.",
        "tier": 0,
    },
    {
        "date": "2019-07-06",
        "person": "Jeffrey Epstein",
        "event": "Epstein arrested at Teterboro Airport on federal sex trafficking charges.",
        "tier": 0,
    },
    {
        "date": "2019-07-12",
        "person": "Alexander Acosta",
        "event": "Alexander Acosta resigns as U.S. Secretary of Labor amid scrutiny of the plea deal.",
        "tier": 2,
    },
    {
        "date": "2019-08-10",
        "person": "Jeffrey Epstein",
        "event": "Jeffrey Epstein found dead in his cell at MCC New York.",
        "tier": 0,
    },
    {
        "date": "2020-07-02",
        "person": "Ghislaine Maxwell",
        "event": "Ghislaine Maxwell arrested in Bradford, New Hampshire.",
        "tier": 0,
    },
    {
        "date": "2021-12-29",
        "person": "Ghislaine Maxwell",
        "event": "Ghislaine Maxwell convicted on five federal counts including sex trafficking.",
        "tier": 0,
    },
    {
        "date": "2022-02-04",
        "person": "Jean-Luc Brunel",
        "event": "Jean-Luc Brunel found dead in his cell in Paris; charged with rape of minors.",
        "tier": 0,
    },
    {
        "date": "2022-02-15",
        "person": "Prince Andrew",
        "event": "Prince Andrew reaches financial settlement with Virginia Giuffre.",
        "tier": 1,
    },
    {
        "date": "2022-06-28",
        "person": "Ghislaine Maxwell",
        "event": "Ghislaine Maxwell sentenced to 20 years in federal prison.",
        "tier": 0,
    },
    {
        "date": "2024-01-03",
        "person": "Multiple",
        "event": "Court unseals documents naming numerous Epstein associates and accusers.",
        "tier": 2,
    },
]


# ---------------------------------------------------------------------------
# Startup / lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Index the corpus on startup if needed."""
    try:
        pipeline = get_pipeline()
        pipeline.index_corpus()
    except Exception as exc:
        log.warning("RAG indexing failed on startup: %s", exc)
    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Paper Trail API",
    description="NLP system for classifying legal consequences in the Epstein case.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/people")
def get_people() -> list[dict]:
    """Return all labeled individuals with their consequence tiers."""
    return get_all_labeled_persons()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Accept a natural language query and return a RAG-generated response.

    Args:
        request: ChatRequest with 'query' field.

    Returns:
        ChatResponse with 'answer' and 'sources'.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    pipeline = get_pipeline()
    try:
        result = pipeline.query(request.query)
    except Exception as exc:
        log.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail="RAG pipeline error.") from exc

    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.get("/timeline")
def get_timeline() -> list[dict]:
    """Return timeline events sorted by date ascending."""
    return sorted(TIMELINE_EVENTS, key=lambda e: e["date"])


@app.get("/person/{name}")
def get_person(name: str) -> dict:
    """Return profile card data for a specific named individual.

    Args:
        name: Full or partial name of the person.

    Returns:
        Dict with name, tier, label, color, and related timeline events.
    """
    profile = predict_for_person(name)
    related_events = [
        e for e in TIMELINE_EVENTS
        if name.lower() in e.get("person", "").lower()
        or e.get("person", "").lower() in name.lower()
    ]
    profile["timeline_events"] = sorted(related_events, key=lambda e: e["date"])
    return profile
