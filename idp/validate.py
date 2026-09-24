"""Validation and routing: decide which fields can be written automatically and which go to a person.

A field is auto-accepted only if it passes its schema rule AND its confidence clears the threshold learned in
cross-validation (the lowest confidence at which auto-accepted values were >= 95% correct). Anything else is
queued for human review with the reasons attached.
"""
from __future__ import annotations

import datetime as dt

MAX_TOTAL = 100_000.0


def _rules(field: str, value, today: dt.date) -> list[str]:
    if value in (None, ""):
        return ["missing"]
    v = str(value)
    if field == "company":
        letters = sum(c.isalpha() for c in v)
        if letters < 3 or letters / len(v) < 0.5:
            return ["doesn't look like a name"]
    elif field == "address":
        if len(v) < 10:
            return ["too short for an address"]
        if not any(c.isdigit() for c in v):
            return ["no street number or postcode"]
    elif field == "date":
        try:
            d = dt.date.fromisoformat(v)
        except ValueError:
            return ["not a valid date"]
        if d.year < 2000:
            return ["date before 2000"]
        if d > today:
            return ["date in the future"]
    elif field == "total":
        try:
            x = float(v)
        except ValueError:
            return ["not a number"]
        if not 0 < x < MAX_TOTAL:
            return ["amount out of range"]
    return []


def validate(fields: dict, thresholds: dict, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    checks = {}
    for name, f in fields.items():
        reasons = _rules(name, f.get("value"), today)
        t = thresholds.get(name, 1.01)
        if not reasons and f.get("conf", 0) < t:
            reasons.append(f"low confidence ({f.get('conf', 0):.2f} < {t:.2f})")
        checks[name] = {"ok": not reasons, "reasons": reasons}
    review = [n for n, c in checks.items() if not c["ok"]]
    return {"checks": checks, "route": "review" if review else "auto", "review_fields": review}
