"""Vercel serverless endpoint for the demo.

GET  /api/extract?meta=1      -> benchmark results and a summary of the 200 test receipts
GET  /api/extract?id=123      -> one test receipt: OCR lines, extracted fields, validation, ground truth
POST /api/extract {ocr pass}  -> run extraction + validation on OCR words produced in the browser (Tesseract.js)

Vercel functions can't run the Tesseract binary, so uploads are OCR'd in the browser with Tesseract.js (the same
engine compiled to WebAssembly) and only the words are sent here. That's a single OCR pass, so the single-pass
models are used; the 4-pass pipeline runs in the AWS Lambda container.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from idp.pipeline import process_ocr  # noqa: E402

MAX_WORDS = 3000


@lru_cache(maxsize=1)
def preds() -> dict:
    return json.loads((ROOT / "results" / "test_predictions.json").read_text())


def meta() -> dict:
    res = json.loads((ROOT / "results" / "results.json").read_text())
    docs = [{"id": i, "route": p["route"], "ok": p["ok"], "review_fields": p["review_fields"],
             "company": p["fields"]["company"]["value"] or p["gold"]["company"]} for i, p in sorted(preds().items())]
    return {"results": res, "docs": docs}


def upload(body: dict) -> tuple[int, dict]:
    words = body.get("words")
    if not isinstance(words, list) or not words:
        return 400, {"error": "send {width, height, words: [{t, x, y, w, h, c, line}]}"}
    if len(words) > MAX_WORDS:
        return 413, {"error": "too many words for a receipt"}
    try:
        clean = [{"t": str(w["t"])[:80], "x": int(w["x"]), "y": int(w["y"]), "w": int(w["w"]), "h": int(w["h"]),
                  "c": float(w.get("c", 0)), "line": [int(v) for v in w["line"]][:3]} for w in words if str(w.get("t", "")).strip()]
    except (KeyError, TypeError, ValueError):
        return 400, {"error": "malformed word entry"}
    doc = {"id": "upload", "width": int(body.get("width", 0)), "height": int(body.get("height", 0)),
           "passes": {"browser": {"ms": int(body.get("ms", 0)), "words": clean}}}
    return 200, process_ocr(doc)


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict, cache: bool = False):
        data = json.dumps(body, separators=(",", ":"), default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "public, s-maxage=86400, max-age=600" if cache else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        if qs.get("meta"):
            return self._send(200, meta(), cache=True)
        doc_id = (qs.get("id") or [""])[0]
        p = preds().get(doc_id)
        if p is None:
            return self._send(404, {"error": "unknown receipt id"})
        return self._send(200, {"id": doc_id, **p}, cache=True)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_500_000:
                return self._send(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "send JSON"})
        status, out = upload(body)
        return self._send(status, out)
