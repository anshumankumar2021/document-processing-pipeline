"""End-to-end processing of one document: OCR -> extraction -> validation -> routing."""
from __future__ import annotations

import time
from functools import lru_cache

from idp.extract import MODEL_PATH, Extractor
from idp.validate import validate


@lru_cache(maxsize=1)
def extractor() -> Extractor:
    return Extractor(MODEL_PATH)


def process_ocr(doc: dict, today=None) -> dict:
    ex = extractor()
    t = time.perf_counter()
    out = ex.extract(doc)
    mode = out["mode"]
    v = validate(out["fields"], ex.thresholds.get(mode, {}), today=today)
    out.update(v)
    out["extract_ms"] = round((time.perf_counter() - t) * 1000, 1)
    out["ocr_ms"] = sum(p.get("ms", 0) for p in doc["passes"].values())
    return out


def process_image(image, doc_id: str = "upload", today=None) -> dict:
    from idp.ocr import run_ocr

    return process_ocr(run_ocr(image, doc_id), today=today)


def record(doc_id: str, result: dict) -> dict:
    """The row written to the results store: values, confidences and routing, no raw OCR."""
    return {"doc_id": doc_id, "route": result["route"], "review_fields": result["review_fields"],
            "fields": {k: {"value": f.get("value"), "conf": f.get("conf"),
                           "ok": result["checks"][k]["ok"], "reasons": result["checks"][k]["reasons"]}
                       for k, f in result["fields"].items()},
            "ocr_ms": result.get("ocr_ms"), "extract_ms": result.get("extract_ms")}
