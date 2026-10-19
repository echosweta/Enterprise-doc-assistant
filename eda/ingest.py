"""Parse documents and split them into section-aware chunks with metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    title: str
    section: str
    department: str = "general"
    version: str = "1"
    is_latest: bool = True
    allowed_roles: list[str] = field(default_factory=lambda: ["all"])
    effective_date: str = ""

    def to_context(self) -> str:
        return f"{self.title} > {self.section}\n{self.text}"


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Tiny front-matter parser (key: value lines between --- fences)."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 4:].lstrip("\n")


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if suffix == ".docx":
        import docx
        out = []
        for p in docx.Document(str(path)).paragraphs:
            style = (p.style.name or "").lower()
            if style.startswith("heading") and p.text.strip():
                level = "".join(c for c in style if c.isdigit()) or "1"
                out.append("#" * int(level) + " " + p.text.strip())
            else:
                out.append(p.text)
        return "\n".join(out)
    raise ValueError(f"Unsupported file type: {path}")


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            pieces.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        pieces.append(cur)
    return pieces


def chunk_document(doc_id: str, raw: str) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    title = meta.get("title", doc_id.replace("_", " ").title())
    roles = [r.strip() for r in meta.get("allowed_roles", "all").split(",") if r.strip()]
    base = dict(
        doc_id=doc_id,
        title=title,
        department=meta.get("department", "general"),
        version=meta.get("version", "1"),
        is_latest=meta.get("is_latest", "true").lower() == "true",
        allowed_roles=roles,
        effective_date=meta.get("effective_date", ""),
    )

    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if buf and "".join(buf).strip():
                sections.append((" > ".join(path) or title, buf))
            level, heading = len(m.group(1)), m.group(2).strip()
            path = path[: level - 1] + [heading]
            buf = []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((" > ".join(path) or title, buf))

    chunks: list[Chunk] = []
    for section, lines in sections:
        text = "\n".join(lines).strip()
        for piece in _split_long(text, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP):
            chunks.append(Chunk(id=f"{doc_id}::{len(chunks)}", text=piece, section=section, **base))
    return chunks


def load_corpus(docs_dir: Path | None = None) -> list[Chunk]:
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in (".md", ".txt", ".pdf", ".docx"):
            chunks.extend(chunk_document(path.stem, read_file(path)))
    return chunks
