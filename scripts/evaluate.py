"""Score the pipeline on the 200 held-out receipts and write results/results.json.

    python -m scripts.evaluate

Reports, for 4-pass OCR voting ("multi") and a single OCR pass ("single"):
  - field accuracy (exact, and near-match for company/address)
  - routing: share of fields and documents auto-accepted, and how accurate the auto-accepted ones are
  - errors reaching the database: wrong values written if everything were auto-accepted, versus only the
    auto-accepted ones (fields sent to review are assumed corrected by the reviewer)
  - OCR ceiling: how often any OCR pass produced the right value at all
Also writes results/test_predictions.json for the demo.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from idp.extract import FIELDS, date_total_candidates, ocr_lines
from idp.features import company_candidates
from idp.pipeline import process_ocr
from idp.score import correct
from idp.text import norm
from scripts.train import load, single

ROOT = Path(__file__).resolve().parent.parent
TODAY = dt.date(2026, 9, 1)   # fixed so re-runs are identical


def pct(x):
    return round(100 * x, 1)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def ceiling(doc, gold):
    """Did any OCR pass produce the right value at all (before vendor matching)?"""
    lines = ocr_lines(doc)
    dt_ = date_total_candidates(lines)
    out = {f: any(correct(f, c["value"], gold[f]) for c in dt_[f]) for f in ("date", "total")}
    out["company"] = any(correct("company", c["value"], gold["company"]) for ls in lines.values() for c in company_candidates(ls))
    out["address"] = any(norm(gold["address"]) and norm(gold["address"]) in norm(" ".join(l.text for l in ls))
                         for ls in lines.values())
    return out


def evaluate(mode: str, ids, labels, docs, known: set):
    rows = []
    for i in ids:
        d = docs[i] if mode == "multi" else single(docs[i])
        r = process_ocr(d, today=TODAY)
        ok = {f: correct(f, r["fields"][f]["value"], labels[i][f]) for f in FIELDS}
        near = {f: correct(f, r["fields"][f]["value"], labels[i][f], near=True) for f in ("company", "address")}
        rows.append({"id": i, "r": r, "ok": ok, "near": near, "ceiling": ceiling(d, labels[i]),
                     "known": norm(labels[i]["company"]) in known})
    n = len(rows)
    by_vendor = {}
    for name, sel in (("known_vendor", [x for x in rows if x["known"]]), ("new_vendor", [x for x in rows if not x["known"]])):
        by_vendor[name] = {"n": len(sel), **{f: pct(sum(x["ok"][f] for x in sel) / len(sel)) if sel else None for f in FIELDS},
                           "docs_straight_through": pct(sum(x["r"]["route"] == "auto" for x in sel) / len(sel)) if sel else None}
    acc = {f: pct(sum(x["ok"][f] for x in rows) / n) for f in FIELDS}
    near = {f: pct(sum(x["near"][f] for x in rows) / n) for f in ("company", "address")}
    fields_total = n * len(FIELDS)
    accepted = [(x["ok"][f]) for x in rows for f in FIELDS if x["r"]["checks"][f]["ok"]]
    wrong_all = sum(not x["ok"][f] for x in rows for f in FIELDS)
    wrong_auto = sum(not x["ok"][f] for x in rows for f in FIELDS if x["r"]["checks"][f]["ok"])
    auto_docs = [x for x in rows if x["r"]["route"] == "auto"]
    per_field = {}
    for f in FIELDS:
        acc_f = [x["ok"][f] for x in rows if x["r"]["checks"][f]["ok"]]
        per_field[f] = {"auto_rate": pct(len(acc_f) / n), "auto_precision": pct(sum(acc_f) / len(acc_f)) if acc_f else None,
                        "wrong_caught": sum(1 for x in rows if not x["ok"][f] and not x["r"]["checks"][f]["ok"]),
                        "wrong_total": sum(1 for x in rows if not x["ok"][f])}
    ocr_ms = [x["r"]["ocr_ms"] for x in rows]
    ext_ms = [x["r"]["extract_ms"] for x in rows]
    res = {
        "n_docs": n,
        "accuracy": acc, "near_match": near,
        "all_fields_correct": pct(sum(all(x["ok"].values()) for x in rows) / n),
        "by_vendor": by_vendor,
        "ocr_ceiling": {f: pct(sum(x["ceiling"][f] for x in rows) / n) for f in FIELDS},
        "routing": {
            "fields_auto_accepted": pct(len(accepted) / fields_total),
            "auto_accepted_precision": pct(sum(accepted) / len(accepted)) if accepted else None,
            "docs_straight_through": pct(len(auto_docs) / n),
            "straight_through_docs_all_correct": pct(sum(all(x["ok"].values()) for x in auto_docs) / len(auto_docs)) if auto_docs else None,
            "wrong_fields_without_review": wrong_all,
            "wrong_fields_written_with_review": wrong_auto,
            "error_reduction": pct(1 - wrong_auto / wrong_all) if wrong_all else None,
            "per_field": per_field,
        },
        "latency_ms": {"ocr_p50": q(ocr_ms, 50), "ocr_p95": q(ocr_ms, 95),
                       "extract_p50": q(ext_ms, 50), "extract_p95": q(ext_ms, 95)},
    }
    return res, rows


def main():
    labels, split, docs = load()
    spec = json.loads((ROOT / "models" / "fields.json").read_text())
    test = split["test"]
    out = {"dataset": "SROIE 2019 (ICDAR) scanned receipts", "n_train": len(split["train"]), "n_test": len(test),
           "ocr": "Tesseract 5, passes " + ", ".join(docs[test[0]]["passes"]), "target_precision": spec["target_precision"],
           "thresholds": spec["thresholds"], "cv_accuracy": spec["cv_accuracy"], "modes": {}}
    known = {norm(labels[i]["company"]) for i in split["train"]}
    preds = {}
    for mode in ("multi", "single"):
        res, rows = evaluate(mode, test, labels, docs, known)
        out["modes"][mode] = res
        print(mode, json.dumps({k: res[k] for k in ("accuracy", "near_match", "all_fields_correct", "ocr_ceiling", "by_vendor")}))
        print("   routing", json.dumps({k: v for k, v in res["routing"].items() if k != "per_field"}))
        if mode == "multi":
            for x in rows:
                r = x["r"]
                preds[x["id"]] = {"gold": labels[x["id"]], "ok": x["ok"], "near": x["near"], "route": r["route"],
                                  "known_vendor": x["known"],
                                  "review_fields": r["review_fields"], "checks": r["checks"],
                                  "fields": {f: {k: v for k, v in r["fields"][f].items()} for f in FIELDS},
                                  "lines": r["lines"], "size": [docs[x["id"]]["width"], docs[x["id"]]["height"]],
                                  "ocr_ms": r["ocr_ms"]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / "test_predictions.json").write_text(json.dumps(preds, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
