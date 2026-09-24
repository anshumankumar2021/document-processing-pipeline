"""OCR: several Tesseract passes per image, each turned into text lines.

One pass misreads different characters than another, so the extractor pools candidates from all passes and uses
agreement between them as evidence (OCR voting). Passes:
    fast6  default eng model, --psm 6 (one uniform block of text)
    best6  tessdata_best eng model, --psm 6
    fast4  default model, --psm 4 (single column of variable-size text)
    best4  tessdata_best model, --psm 4

Document format (also the cache format, one per line in data/ocr.jsonl.gz):
    {"id", "width", "height", "passes": {name: {"ms", "words": [{"t", "x", "y", "w", "h", "c", "line": [b, p, l]}]}}}
A single-pass document (e.g. words from Tesseract.js in the browser) is just a "passes" dict with one entry.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import os
from pathlib import Path

BEST_DIR = Path(os.environ.get("TESSDATA_BEST", Path(__file__).resolve().parent.parent / "data" / "tessdata"))
PASSES = {"fast6": ("fast", 6), "best6": ("best", 6), "fast4": ("fast", 4), "best4": ("best", 4)}


@dataclass
class Line:
    idx: int
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float                 # mean word confidence, 0-100
    height: float               # median word height
    words: list = field(default_factory=list)


def run_pass(image, name: str) -> dict:
    import pytesseract

    model, psm = PASSES[name]
    cfg = f"--psm {psm}" + (f" --tessdata-dir {BEST_DIR}" if model == "best" else "")
    im = image.convert("L")
    t = time.perf_counter()
    d = pytesseract.image_to_data(im, config=cfg, output_type=pytesseract.Output.DICT)
    ms = round((time.perf_counter() - t) * 1000)
    words = [{"t": txt, "x": d["left"][i], "y": d["top"][i], "w": d["width"][i], "h": d["height"][i],
              "c": round(float(d["conf"][i]), 1), "line": [d["block_num"][i], d["par_num"][i], d["line_num"][i]]}
             for i, txt in enumerate(d["text"]) if txt.strip()]
    return {"ms": ms, "words": words}


def run_ocr(image, doc_id: str = "upload", passes=tuple(PASSES)) -> dict:
    return {"id": doc_id, "width": image.width, "height": image.height,
            "passes": {p: run_pass(image, p) for p in passes}}


def to_lines(ocr_pass: dict) -> list[Line]:
    groups: dict[tuple, list] = {}
    for w in ocr_pass["words"]:
        groups.setdefault(tuple(w["line"]), []).append(w)
    lines = []
    for ws in groups.values():
        ws.sort(key=lambda w: w["x"])
        text = " ".join(w["t"] for w in ws)
        lines.append(Line(idx=0, text=text, x0=min(w["x"] for w in ws), y0=min(w["y"] for w in ws),
                          x1=max(w["x"] + w["w"] for w in ws), y1=max(w["y"] + w["h"] for w in ws),
                          conf=statistics.mean(max(w["c"], 0) for w in ws),
                          height=statistics.median(w["h"] for w in ws), words=ws))
    lines.sort(key=lambda l: (l.y0, l.x0))
    for i, l in enumerate(lines):
        l.idx = i
    return lines
