"""Run the four Tesseract passes over every receipt and pack the words into data/ocr.jsonl.gz.

    python -m scripts.run_ocr            # all receipts missing from the cache
    python -m scripts.run_ocr --check 25 # re-OCR 25 receipts and fail unless they match the cache word for word
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # one thread per pass; parallelise across receipts instead

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "sroie" / "data" / "img"


def ocr(doc_id: str) -> dict:
    from PIL import Image

    from idp.ocr import run_ocr
    return run_ocr(Image.open(IMG / f"{doc_id}.jpg"), doc_id)


def words(doc):
    return {p: [w["t"] for w in v["words"]] for p, v in doc["passes"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, help="re-OCR this many receipts and compare with the cache")
    args = ap.parse_args()
    ids = sorted(p.stem for p in IMG.glob("*.jpg"))
    if args.check:
        sample = ids[:: max(1, len(ids) // args.check)][: args.check]
        with Pool(os.cpu_count() or 2) as pool:
            fresh = dict(zip(sample, pool.map(ocr, sample)))
        from idp.ocrcache import get
        diff = [i for i in sample if words(fresh[i]) != words(get(i))]
        print(f"{len(sample) - len(diff)}/{len(sample)} receipts re-OCR'd identically")
        if diff:
            raise SystemExit(f"OCR differs from the cache for {diff}")
        return
    from idp.ocrcache import PACK, load_all, write_all
    docs = dict(load_all()) if PACK.exists() else {}
    todo = [i for i in ids if i not in docs]
    with Pool(os.cpu_count() or 2) as pool:
        for k, doc in enumerate(pool.imap_unordered(ocr, todo)):
            docs[doc["id"]] = doc
            if k % 50 == 0:
                print(k, doc["id"], flush=True)
                write_all(docs)   # checkpoint
    write_all(docs)
    print(f"{len(docs)} receipts in {PACK}")


if __name__ == "__main__":
    main()
