"""Vendor master data: names and branch addresses of vendors seen before.

Real document-processing systems match an OCR'd vendor name against the vendors they already know and use the
canonical name and address, because OCR rarely gets a long name exactly right. Here the vendor book is built from
the *training* receipts' labels only; a receipt from a vendor that isn't in the book falls back to the OCR text.
"""
from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from idp.text import norm

BOOK_PATH = Path(__file__).resolve().parent.parent / "models" / "vendors.json"


def sim(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(norm(a), norm(b))


def name_score(cand: str, name: str, **_) -> float:
    """Similarity of an OCR line to a vendor name, tolerating extra text on the line (e.g. a registration number)."""
    s = Levenshtein.normalized_similarity(cand, name)
    if len(name) >= 8 and len(name) <= len(cand) <= 1.6 * len(name):
        s = max(s, fuzz.partial_ratio(name, cand) / 100 * 0.97)
    return s


class VendorBook:
    def __init__(self, entries: dict[str, dict] | None = None):
        # {normalised name: {"name": canonical, "addresses": {normalised: canonical}}}
        self.entries = entries or {}
        self._keys = list(self.entries)

    @classmethod
    def from_labels(cls, labels: dict, ids) -> "VendorBook":
        e: dict[str, dict] = {}
        for i in ids:
            name, addr = labels[i]["company"], labels[i]["address"]
            if not name:
                continue
            v = e.setdefault(norm(name), {"name": name, "addresses": {}})
            if addr:
                v["addresses"].setdefault(norm(addr), addr)
        return cls(e)

    @classmethod
    def load(cls, path: Path = BOOK_PATH) -> "VendorBook":
        return cls(json.loads(Path(path).read_text()) if Path(path).exists() else {})

    def save(self, path: Path = BOOK_PATH):
        Path(path).write_text(json.dumps(self.entries, indent=0, ensure_ascii=False))

    def match_name(self, text: str) -> tuple[dict | None, float]:
        if not self._keys or not text:
            return None, 0.0
        hit = process.extractOne(norm(text), self._keys, scorer=name_score)
        return (self.entries[hit[0]], float(hit[1])) if hit else (None, 0.0)

    def address_matches(self, page_text: str, min_evidence: float = 0.85) -> list[tuple[dict, float]]:
        """Vendors whose known branch address is printed on this receipt, even if the name itself (often a logo)
        wasn't read."""
        page = norm(page_text)
        out = []
        for e in self.entries.values():
            ev = max((fuzz.partial_ratio(a, page) / 100 for a in e["addresses"] if len(a) >= 20), default=0.0)
            if ev >= min_evidence:
                out.append((e, ev))
        return out

    @staticmethod
    def address_evidence(address: str, ocr_addresses: list[str], page_text: str) -> float:
        """How well a known address is supported by this receipt's OCR (0-1)."""
        best = max((sim(address, a) for a in ocr_addresses), default=0.0)
        return max(best, fuzz.partial_ratio(norm(address), norm(page_text)) / 100 * 0.98)
