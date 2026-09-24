"""Train the field scorers, build the vendor book, and choose review thresholds.

    python -m scripts.train

1. Vendor book from the training labels (names and branch addresses). While building training examples each
   receipt is left out of its own book, so a vendor seen only once looks new, as it would in production.
2. One logistic regression per field (scikit-learn), in order: company; address line tagger (fed the *predicted*
   company line); address chooser (OCR address vs known branch addresses); date; total. Weights are exported to
   models/fields.json for dependency-light inference. Two sets: "multi" (4 OCR passes) and "single" (1 pass).
3. 5-fold cross-validation over the training receipts gives out-of-fold confidences; for each field pick the lowest
   threshold at which auto-accepted values are >= TARGET_PRECISION correct. The test split is never used here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression

from idp.extract import (FIELDS, Extractor, LogReg, address_cands, company_cands, date_total_candidates,
                         ocr_addresses, ocr_lines)
from idp.features import address_line_feats
from idp.ocrcache import load_all
from idp.score import correct, similarity
from idp.text import norm
from idp.vendors import VendorBook

ROOT = Path(__file__).resolve().parent.parent
TARGET_PRECISION = 0.95
C = 1.0


def load():
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    split = json.loads((ROOT / "data" / "split.json").read_text())
    cache = load_all()
    docs = {i: cache[i] for i in labels}
    return labels, split, docs


def single(doc: dict) -> dict:
    """The same receipt with only the default OCR pass (what the browser upload path has)."""
    return {**doc, "passes": {"fast6": doc["passes"]["fast6"]}}


def fit_one(X: list[dict], y: list[int]) -> dict:
    names = sorted({k for f in X for k in f})
    A = np.array([[f.get(n, 0.0) for n in names] for f in X])
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=C, max_iter=5000).fit((A - mu) / sd, y)
    return {"features": names, "mean": mu.round(6).tolist(), "scale": sd.round(6).tolist(),
            "coef": m.coef_[0].round(6).tolist(), "intercept": round(float(m.intercept_[0]), 6),
            "n_examples": len(y), "n_positive": int(sum(y))}


def fit_set(ids, labels, docs) -> dict:
    books = {i: VendorBook.from_labels(labels, [j for j in ids if j != i]) for i in ids}
    prep = {i: ocr_lines(docs[i]) for i in ids}
    ccs = {i: company_cands(prep[i], books[i]) for i in ids}
    X, y = [], []
    for i in ids:
        g = labels[i]["company"]
        exact = [correct("company", c["value"], g) for c in ccs[i]]
        if any(exact):
            ys = exact
        else:   # no exact candidate (OCR error, new vendor): teach the layout with the closest line instead
            sims = [similarity(c["value"], g) for c in ccs[i]]
            top = max(sims, default=0)
            ys = [s == top and top >= 0.8 for s in sims]
        for c, t in zip(ccs[i], ys):
            X.append(c["feats"]); y.append(int(t))
    company = LogReg(spec_company := fit_one(X, y))

    # address line tagger, fed the predicted company line of each pass
    company_by = {}
    Xt, yt = [], []
    for i in ids:
        for c in ccs[i]:
            c["p"] = company.prob(c["feats"])
        company_by[i] = {p: max((c for c in ccs[i] if c["pass"] == p and c["lines"]), key=lambda c: c["p"], default={"lines": [None]})["lines"][0]
                         for p in prep[i]}
        ga = norm(labels[i]["address"])
        for p, ls in prep[i].items():
            for t in address_line_feats(ls, company_by[i][p]):
                txt = norm(t["text"])
                Xt.append(t["feats"]); yt.append(int(len(txt) >= 4 and bool(ga) and fuzz.partial_ratio(txt, ga) >= 90))
    spec_tagger = fit_one(Xt, yt)
    tagger = LogReg(spec_tagger)

    # address chooser
    Xa, ya = [], []
    for i in ids:
        best = max(ccs[i], key=lambda c: c["p"], default=None)
        vendor = best.get("vendor") if best else None
        for c in address_cands(prep[i], ocr_addresses(prep[i], company_by[i], tagger), vendor):
            Xa.append(c["feats"]); ya.append(int(correct("address", c["value"], labels[i]["address"])))

    Xd, yd, Xs, ys = [], [], [], []
    for i in ids:
        dt = date_total_candidates(prep[i])
        for c in dt["date"]:
            Xd.append(c["feats"]); yd.append(int(correct("date", c["value"], labels[i]["date"])))
        for c in dt["total"]:
            Xs.append(c["feats"]); ys.append(int(correct("total", c["value"], labels[i]["total"])))
    return {"company": spec_company, "address_lines": spec_tagger, "address": fit_one(Xa, ya),
            "date": fit_one(Xd, yd), "total": fit_one(Xs, ys)}


def fit(ids, labels, docs) -> dict:
    return {"models": {"multi": fit_set(ids, labels, docs),
                       "single": fit_set(ids, labels, {i: single(docs[i]) for i in ids})}}


def pick_threshold(confs: list[float], oks: list[bool], target: float) -> float:
    """Lowest threshold whose accepted set is at least `target` correct (1.01 = accept nothing)."""
    pairs = sorted(zip(confs, oks), reverse=True)
    best, n_ok = 1.01, 0
    for k, (c, ok) in enumerate(pairs, 1):
        n_ok += ok
        if n_ok / k >= target:
            best = c
    return round(best, 4)


def main():
    labels, split, docs = load()
    train = split["train"]
    tmp = ROOT / "models" / "_cv.json"
    order = train[:]
    random.Random(7).shuffle(order)
    folds = [order[k::5] for k in range(5)]
    oof = {m: {f: ([], []) for f in FIELDS} for m in ("multi", "single")}
    for k in range(5):
        held = set(folds[k])
        fit_ids = [i for i in train if i not in held]
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text(json.dumps(fit(fit_ids, labels, docs)))
        ex = Extractor(tmp, book=VendorBook.from_labels(labels, fit_ids))
        for i in folds[k]:
            for mode, d in (("multi", docs[i]), ("single", single(docs[i]))):
                out = ex.extract(d)["fields"]
                for f in FIELDS:
                    oof[mode][f][0].append(out[f]["conf"])
                    oof[mode][f][1].append(correct(f, out[f]["value"], labels[i][f]))
    tmp.unlink()
    thresholds = {m: {f: pick_threshold(*oof[m][f], TARGET_PRECISION) for f in FIELDS} for m in oof}
    cv_acc = {m: {f: round(float(np.mean(oof[m][f][1])), 4) for f in FIELDS} for m in oof}

    spec = fit(train, labels, docs)
    spec.update({"thresholds": thresholds, "target_precision": TARGET_PRECISION, "cv_accuracy": cv_acc,
                 "trained_on": len(train)})
    (ROOT / "models" / "fields.json").write_text(json.dumps(spec, indent=1))
    VendorBook.from_labels(labels, train).save(ROOT / "models" / "vendors.json")
    print("5-fold CV accuracy on train:", json.dumps(cv_acc))
    print("review thresholds:", json.dumps(thresholds))


if __name__ == "__main__":
    main()
