# Enterprise Document Assistant (RAG)

Internal Q&A over company documents (HR, IT, compliance, SOPs) with hybrid retrieval, role-based access,
version awareness, cited answers and a confidence gate that routes to a human instead of guessing.

```
docs -> parse -> section-aware chunks + metadata -> embeddings + BM25 index
question -> role/version filter -> dense + BM25 -> RRF -> (rerank) -> confidence gate -> Claude -> answer + citations
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && export $(cat .env | xargs)      # add ANTHROPIC_API_KEY
python -m eval.run_eval --sweep --rebuild               # build index + see metrics
uvicorn eda.api:app --reload                            # API on :8000
streamlit run ui/streamlit_app.py                       # UI on :8501
pytest -q
```
Without `ANTHROPIC_API_KEY` the app returns an extractive answer, so everything runs offline.
Without `sentence-transformers` it falls back to a hashing embedder (set `FORCE_HASH_EMBEDDINGS=1` to force it).

## Project layout
| Path | Purpose |
|---|---|
| `eda/ingest.py` | md/pdf/docx parsing, heading-aware chunking, metadata |
| `eda/index.py` | embedder, BM25, index persistence |
| `eda/retrieve.py` | access/version filter, hybrid search, RRF, rerank, confidence |
| `eda/generate.py` | grounded prompt, citations, injection-resistant framing |
| `eda/pipeline.py` | orchestration + refusal logic |
| `eda/api.py` | FastAPI: `/ask`, `/feedback`, `/health` |
| `eval/run_eval.py` | recall@k, MRR, refusal rate, leak check, threshold sweep |
| `data/docs`, `data/eval` | synthetic company docs and 32 labelled questions |

## Next steps (in order)
1. Install `sentence-transformers`, run `python -m eval.run_eval --rerank --sweep --rebuild`, and **recalibrate `CONFIDENCE_THRESHOLD`** (embedding scales differ).
2. Add 20+ harder questions to `data/eval/questions.json` (multi-doc, ambiguous, version conflicts).
3. Add an LLM-as-judge faithfulness score to `eval/run_eval.py`.
4. Swap the pickle index for Qdrant or pgvector; add an ingestion endpoint.
5. Derive `role` from SSO/JWT claims; log queries, latency, token cost.
6. Deploy (Render / Fly.io / VM) with the Dockerfile and record a demo GIF for the README.

## Results (fill in after running with real embeddings)
| Retrieval | recall@5 | MRR |
|---|---|---|
| BM25 | | |
| Dense | | |
| Hybrid | | |
| Hybrid + rerank | | |
# Enterprise-doc-assistant
