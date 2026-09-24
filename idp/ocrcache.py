"""The OCR cache: every receipt's four Tesseract passes, packed into data/ocr.jsonl.gz (one JSON document per line).

Re-running OCR takes ~10 minutes on two cores, so training, evaluation and tests read this cache instead.
CI re-OCRs a sample of receipts and checks they match it word for word (scripts/run_ocr.py --check).
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent / "data" / "ocr.jsonl.gz"


@lru_cache(maxsize=1)
def load_all() -> dict:
    with gzip.open(PACK, "rt", encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f)}


def get(doc_id: str) -> dict:
    return load_all()[doc_id]


def write_all(docs: dict):
    with gzip.open(PACK, "wt", encoding="utf-8", compresslevel=9) as f:
        for k in sorted(docs):
            f.write(json.dumps(docs[k], separators=(",", ":"), ensure_ascii=False) + "\n")
