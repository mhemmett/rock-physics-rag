#!/usr/bin/env python3
"""
Extracts and chunks full-text from PDFs in catalog/pdfs/.
Outputs catalog/pdf-chunks.json, which build-index.mjs picks up automatically.

Usage:
    pip install pymupdf
    python3 scripts/chunk-pdfs.py

Drop any PDF into catalog/pdfs/ and re-run. If a PDF's title matches an
existing Zotero entry in catalog/papers.json, it inherits that paper's
metadata (DOI, year, journal, author). Otherwise metadata is inferred from
the PDF itself.

build-index.mjs skips the abstract-only Zotero chunk for any paper that has
full-text chunks here, so you don't get duplicate coverage.
"""

import json
import re
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    print("pymupdf not installed.  Run: pip install pymupdf")
    sys.exit(1)

ROOT    = Path(__file__).parent.parent
PDF_DIR = ROOT / "catalog" / "pdfs"
OUT     = ROOT / "catalog" / "pdf-chunks.json"
PAPERS  = ROOT / "catalog" / "papers.json"

CHUNK_WORDS   = 400   # target words per chunk
OVERLAP_WORDS = 50    # words of overlap between consecutive chunks


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)          # fix hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)           # collapse blank lines
    text = re.sub(r"[ \t]+", " ", text)              # normalise spaces
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        # Drop likely headers/footers: short all-caps lines or page numbers
        if s and len(s.split()) <= 3 and (s.isupper() or re.fullmatch(r"\d+", s)):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def split_chunks(text: str) -> list[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        end = min(i + CHUNK_WORDS, len(words))
        out.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += CHUNK_WORDS - OVERLAP_WORDS
    return out


# ── Metadata resolution ───────────────────────────────────────────────────────

def load_papers_lookup() -> dict:
    """Build title→paper and doi→paper dicts from catalog/papers.json."""
    if not PAPERS.exists():
        return {}
    data = json.loads(PAPERS.read_text())
    lookup = {}
    for p in data.get("papers", []):
        if p.get("doi"):
            lookup[p["doi"].lower()] = p
        if p.get("title"):
            lookup[p["title"].lower()] = p
    return lookup


def resolve_meta(doc, pdf_path: Path, lookup: dict) -> dict:
    raw   = doc.metadata or {}
    title = raw.get("title", "").strip()
    author = raw.get("author", "").strip()

    # Try to match against Zotero papers by title
    match = lookup.get(title.lower()) if title else None

    if match:
        return {
            "title":        match["title"],
            "doi":          match.get("doi"),
            "year":         match.get("year"),
            "journal":      match.get("journal"),
            "first_author": match.get("first_author"),
            "tags":         match.get("tags", []),
        }

    # Fallback: use PDF metadata / filename
    if not title:
        title = pdf_path.stem.replace("-", " ").replace("_", " ").title()

    first_author = ""
    if author:
        first_author = author.split(";")[0].split(",")[0].strip()

    return {
        "title":        title,
        "doi":          None,
        "year":         None,
        "journal":      None,
        "first_author": first_author or None,
        "tags":         [],
    }


# ── Per-PDF processing ────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, lookup: dict) -> list[dict]:
    print(f"  {pdf_path.name}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    ERROR: {e}")
        return []

    text = clean("\n\n".join(page.get_text() for page in doc))
    doc_meta = resolve_meta(doc, pdf_path, lookup)
    doc.close()

    word_count = len(text.split())
    if word_count < 100:
        print(f"    WARNING: only {word_count} words extracted — scanned/image PDF?")
        return []

    chunks     = split_chunks(text)
    id_base    = (
        f"paper::{doc_meta['doi']}"
        if doc_meta.get("doi")
        else f"paper::pdf::{pdf_path.stem}"
    )

    result = []
    for i, chunk_text in enumerate(chunks):
        header = doc_meta["title"]
        if doc_meta.get("first_author"):
            header += f" — {doc_meta['first_author']}"
        if doc_meta.get("year"):
            header += f" ({doc_meta['year']})"

        result.append({
            "id":           f"{id_base}::chunk{i}",
            "title":        doc_meta["title"],
            "doi":          doc_meta.get("doi"),
            "year":         doc_meta.get("year"),
            "journal":      doc_meta.get("journal"),
            "first_author": doc_meta.get("first_author"),
            "tags":         doc_meta.get("tags", []),
            "chunk_index":  i,
            "chunk_total":  len(chunks),
            "text":         chunk_text,
        })

    print(f"    → {len(chunks)} chunks  ({word_count} words)")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    lookup = load_papers_lookup()

    # Collect PDFs: manually dropped files + Zotero-linked paths from papers.json
    seen = set()
    pdfs = []

    for p in sorted(PDF_DIR.glob("*.pdf")):
        if p not in seen:
            seen.add(p)
            pdfs.append(p)

    if PAPERS.exists():
        data = json.loads(PAPERS.read_text())
        for paper in data.get("papers", []):
            pdf_path_str = paper.get("pdf_path")
            if not pdf_path_str:
                continue
            p = Path(pdf_path_str)
            if p.exists() and p not in seen:
                seen.add(p)
                pdfs.append(p)

    if not pdfs:
        print("No PDFs found in catalog/pdfs/ and no pdf_path entries in papers.json.")
        print("Run fetch-zotero.py first, or drop PDFs into catalog/pdfs/.")
        OUT.write_text(json.dumps({"chunks": []}, indent=2))
        return

    print(f"Processing {len(pdfs)} PDF(s)\n")
    all_chunks = []
    for pdf_path in pdfs:
        all_chunks.extend(process_pdf(pdf_path, lookup))

    OUT.write_text(json.dumps({"chunks": all_chunks}, indent=2, ensure_ascii=False))
    unique_papers = len(set(c["title"] for c in all_chunks))
    print(f"\n{len(all_chunks)} chunks from {unique_papers} papers → {OUT.relative_to(ROOT)}")
    print("Run next: node scripts/build-index.mjs")


if __name__ == "__main__":
    main()
