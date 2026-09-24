"""How a predicted field is compared with the label.

- total: same amount (currency marks like 'RM' or '$' ignored)
- date: same calendar date (day-first, as printed on Malaysian receipts); falls back to string match
- company, address: exact match after upper-casing and collapsing whitespace ("exact"), and a
  near-match rate (normalised edit similarity >= 0.9) that tolerates a character or two of OCR noise
"""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from idp.text import norm, parse_date, parse_money


def similarity(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(norm(a), norm(b))


def correct(field: str, pred, gold: str, near: bool = False) -> bool:
    if pred in (None, ""):
        return not gold
    if field == "total":
        g = parse_money(gold)
        return g is not None and abs(float(pred) - g) < 0.005
    if field == "date":
        g = parse_date(gold)
        from datetime import date
        try:
            p = date.fromisoformat(pred)
        except (TypeError, ValueError):
            p = parse_date(str(pred))
        return (p == g) if g else norm(str(pred)) == norm(gold)
    if near:
        return similarity(str(pred), gold) >= 0.9
    return norm(str(pred)) == norm(gold)
