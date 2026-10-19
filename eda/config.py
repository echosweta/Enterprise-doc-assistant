import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "data" / "index.pkl"))
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", ROOT / "data" / "feedback.jsonl"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATES = int(os.getenv("CANDIDATES", "20"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
