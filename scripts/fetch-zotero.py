#!/usr/bin/env python3
"""
Exports papers from the Rock Physics Zotero collection to catalog/papers.json.
Reads a copy of ~/Zotero/zotero.sqlite to avoid lock contention with the app.

Run: python3 scripts/fetch-zotero.py
"""

import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

ZOTERO_DB      = Path.home() / "Zotero" / "zotero.sqlite"
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"
OUT = Path(__file__).parent.parent / "catalog" / "papers.json"

TARGET_COLLECTIONS = {"Rock Physics"}


def main():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        shutil.copy2(ZOTERO_DB, tmp.name)
        db_path = tmp.name

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    placeholders = ",".join("?" * len(TARGET_COLLECTIONS))
    collection_ids = [
        row["collectionID"]
        for row in con.execute(
            f"SELECT collectionID FROM collections WHERE collectionName IN ({placeholders})",
            list(TARGET_COLLECTIONS),
        )
    ]
    if not collection_ids:
        raise RuntimeError(f"No collections matching {TARGET_COLLECTIONS}")
    print(f"Found collection IDs: {collection_ids}")

    ph = ",".join("?" * len(collection_ids))
    item_ids = [
        row["itemID"]
        for row in con.execute(
            f"SELECT DISTINCT itemID FROM collectionItems WHERE collectionID IN ({ph})",
            collection_ids,
        )
    ]
    print(f"Found {len(item_ids)} items")

    ph = ",".join("?" * len(item_ids))
    fields_by_item = {}
    for row in con.execute(
        f"""SELECT id.itemID, f.fieldName, idv.value
            FROM itemData id
            JOIN fields f ON id.fieldID = f.fieldID
            JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE id.itemID IN ({ph})
              AND f.fieldName IN ('title','abstractNote','DOI','date','url','publicationTitle')""",
        item_ids,
    ):
        fields_by_item.setdefault(row["itemID"], {})[row["fieldName"]] = row["value"]

    # PDF attachments: linkMode 1 = stored file, contentType = application/pdf
    pdf_by_item = {}
    for row in con.execute(
        f"""SELECT ia.parentItemID, ia.path, i.key
            FROM itemAttachments ia
            JOIN items i ON ia.itemID = i.itemID
            WHERE ia.parentItemID IN ({ph})
              AND ia.contentType = 'application/pdf'
              AND ia.linkMode = 1""",
        item_ids,
    ):
        if row["parentItemID"] in pdf_by_item:
            continue  # keep first attachment only
        filename = row["path"].replace("storage:", "") if row["path"] else ""
        if filename:
            full_path = ZOTERO_STORAGE / row["key"] / filename
            if full_path.exists():
                pdf_by_item[row["parentItemID"]] = str(full_path)

    tags_by_item = {}
    for row in con.execute(
        f"""SELECT it.itemID, t.name
            FROM itemTags it
            JOIN tags t ON it.tagID = t.tagID
            WHERE it.itemID IN ({ph})""",
        item_ids,
    ):
        tags_by_item.setdefault(row["itemID"], []).append(row["name"])

    # First author's last name (orderIndex = 0, creatorType = 'author')
    creators_by_item = {}
    for row in con.execute(
        f"""SELECT ic.itemID, c.lastName, c.firstName, ic.orderIndex, ct.creatorType
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
            WHERE ic.itemID IN ({ph})
              AND ct.creatorType = 'author'
            ORDER BY ic.itemID, ic.orderIndex""",
        item_ids,
    ):
        if row["itemID"] not in creators_by_item:
            creators_by_item[row["itemID"]] = row["lastName"] or row["firstName"] or ""

    con.close()

    papers = []
    for item_id in item_ids:
        f = fields_by_item.get(item_id, {})
        title = f.get("title", "").strip()
        if not title:
            continue
        abstract = f.get("abstractNote", "").strip()
        tags = tags_by_item.get(item_id, [])
        year_match = re.search(r"\b(19|20)\d{2}\b", f.get("date", ""))
        year = int(year_match.group()) if year_match else None
        papers.append({
            "title": title,
            "abstract": abstract,
            "doi": f.get("DOI", "").strip() or None,
            "year": year,
            "journal": f.get("publicationTitle", "").strip() or None,
            "first_author": creators_by_item.get(item_id) or None,
            "tags": tags,
            "linked_instruments": [],
            "pdf_path": pdf_by_item.get(item_id),
        })

    papers.sort(key=lambda p: p["year"] or 0, reverse=True)
    out = {"version": "1.0", "source": "Zotero Rock Physics collection", "papers": papers}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    pdf_count = sum(1 for p in papers if p.get("pdf_path"))
    print(f"Wrote {len(papers)} papers ({pdf_count} with attached PDF) → {OUT}")


if __name__ == "__main__":
    main()
