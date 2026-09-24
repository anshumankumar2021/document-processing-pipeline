"""Parsing helpers for OCR text: dates, money amounts, normalisation."""
from __future__ import annotations

import datetime as dt
import re

MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

# OCR-tolerant date patterns. Each yields (raw match, kind).
_DATE_PATTERNS = [
    ("dmy", re.compile(r"(?<!\d)(\d{1,2})\s?[/\-.]\s?(\d{1,2})\s?[/\-.]\s?(\d{4}|\d{2})(?!\d)")),
    ("ymd", re.compile(r"(?<!\d)(\d{4})\s?[/\-.]\s?(\d{1,2})\s?[/\-.]\s?(\d{1,2})(?!\d)")),
    ("dMy", re.compile(r"(?<!\d)(\d{1,2})[\s\-/.]?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-/.,]*(\d{4}|\d{2})(?!\d)", re.I)),
]

_MONEY = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)\s?[.,]\s?(\d{2})(?![\d])")


def norm(s: str) -> str:
    """Upper-case and collapse whitespace (how company and address are compared)."""
    return " ".join((s or "").upper().split())


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def parse_date_parts(kind: str, a: str, b: str, c: str) -> dt.date | None:
    try:
        if kind == "dmy":
            d, m, y = int(a), int(b), _year(c)
            if m > 12 and d <= 12:           # month-first (US style) receipt
                d, m = m, d
        elif kind == "ymd":
            y, m, d = int(a), int(b), int(c)
        else:
            d, m, y = int(a), MONTHS[b[:3].upper()], _year(c)
        return dt.date(y, m, d)
    except (ValueError, KeyError):
        return None


def find_dates(text: str) -> list[dict]:
    out = []
    for kind, pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            out.append({"raw": m.group(0), "kind": kind, "start": m.start(), "date": parse_date_parts(kind, *m.groups()),
                        "short_year": len(m.group(3 if kind != "ymd" else 1)) == 2})
    return out


def parse_date(s: str) -> dt.date | None:
    """Parse a whole date string (used for ground truth and for validating LLM output)."""
    for d in find_dates(s or ""):
        if d["date"]:
            return d["date"]
    return None


def find_money(text: str) -> list[dict]:
    out = []
    for m in _MONEY.finditer(text):
        whole = m.group(1).replace(",", "")
        out.append({"raw": m.group(0), "value": round(float(f"{whole}.{m.group(2)}"), 2), "start": m.start(),
                    "end": m.end()})
    return out


def parse_money(s: str) -> float | None:
    """Parse a total such as 'RM 9.00', '$1,234.50' or '9.00'."""
    s = (s or "").upper().replace("RM", "").replace("$", "").strip()
    found = find_money(s)
    if found:
        return found[-1]["value"]
    try:
        return round(float(s.replace(",", "")), 2)
    except ValueError:
        return None
