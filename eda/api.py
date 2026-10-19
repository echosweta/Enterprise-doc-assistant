"""FastAPI service. Run: uvicorn eda.api:app --reload"""
from __future__ import annotations

import json
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config
from .pipeline import ask, get_index

app = FastAPI(title="Enterprise Document Assistant")
_index = None


def index():
    global _index
    if _index is None:
        _index = get_index()
    return _index


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    role: str = "employee"
    rerank: bool = False


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool
    comment: str = ""


@app.get("/health")
def health():
    idx = index()
    return {"status": "ok", "chunks": len(idx.chunks), "embedder": idx.embedder_name}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    # NOTE: in production derive `role` from the authenticated user (SSO/JWT), never from the request body.
    return ask(index(), req.question, role=req.role, rerank=req.rerank)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.FEEDBACK_PATH, "a") as f:
        f.write(json.dumps({**req.model_dump(), "ts": time.time()}) + "\n")
    return {"ok": True}
