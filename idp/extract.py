"""Field extraction with OCR voting and vendor matching.

For every OCR pass, generate candidates for each field (features.py). Candidates from all passes are pooled and
each gets an `agree` feature: the share of passes that produced the same value. Company candidates that closely
match a known vendor also produce a *canonical* candidate (the vendor's name as recorded before). A logistic
regression per field picks the winner; its probability is the field's confidence.

Address is built in two steps: a line tagger assembles one address per OCR pass, then a chooser picks between
those and the known branch addresses of the matched vendor.

Two model sets live in models/fields.json: "multi" (4 OCR passes, used by the pipeline) and "single" (one pass,
e.g. OCR done in the browser). Inference needs only rapidfuzz besides the standard library.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from idp.features import address_line_feats, clean, company_candidates, date_candidates, total_candidates
from idp.ocr import to_lines
from idp.text import norm
from idp.vendors import BOOK_PATH, VendorBook, sim

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fields.json"
FIELDS = ("company", "date", "address", "total")
VENDOR_MIN_SIM = 0.75


class LogReg:
    def __init__(self, spec: dict):
        self.names = spec["features"]
        self.mean, self.scale = spec["mean"], spec["scale"]
        self.coef, self.b = spec["coef"], spec["intercept"]

    def prob(self, feats: dict) -> float:
        z = self.b
        for n, mu, sd, w in zip(self.names, self.mean, self.scale, self.coef):
            z += w * ((feats.get(n, 0.0) - mu) / sd)
        return 1 / (1 + math.exp(-max(min(z, 40), -40)))


def key(field: str, cand: dict):
    if field == "date":
        return cand.get("date") or norm(cand["value"])
    if field == "total":
        return cand["num"]
    return norm(cand["value"])


def vote(field: str, cands: list[dict], n_passes: int):
    votes: dict = {}
    for c in cands:
        votes.setdefault(key(field, c), set()).add(c["pass"])
    for c in cands:
        c["feats"]["agree"] = len(votes[key(field, c)]) / n_passes
        c["feats"]["agree_all"] = float(len(votes[key(field, c)]) == n_passes)


def ocr_lines(doc: dict) -> dict:
    return {p: to_lines(v) for p, v in doc["passes"].items()}


def date_total_candidates(lines: dict) -> dict:
    out = {}
    for field, gen in (("date", date_candidates), ("total", total_candidates)):
        cands = [dict(c, **{"pass": p}) for p, ls in lines.items() for c in gen(ls)]
        vote(field, cands, len(lines))
        out[field] = cands
    return out


def company_cands(lines: dict, book: VendorBook) -> list[dict]:
    cands = []
    page = " ".join(l.text for ls in lines.values() for l in ls[:30])
    by_address = {norm(e["name"]): ev for e, ev in book.address_matches(page)}
    for p, ls in lines.items():
        for c in company_candidates(ls):
            c["pass"] = p
            entry, s = book.match_name(c["value"])
            c["feats"].update({"vendor_sim": s, "canonical": 0.0, "addr_evidence": 0.0, "from_address": 0.0,
                               # an OCR line that is *almost* a known vendor is probably that vendor, misread
                               "near_known": float(VENDOR_MIN_SIM <= s < 0.999)})
            cands.append(c)
            if entry and s >= VENDOR_MIN_SIM:
                cands.append({"value": entry["name"], "lines": c["lines"], "pass": p, "vendor": entry,
                              "feats": {**c["feats"], "canonical": 1.0, "exact_vendor": float(s >= 0.999),
                                        "addr_evidence": by_address.get(norm(entry["name"]), 0.0)}})
    named = {norm(c["value"]) for c in cands if c["feats"]["canonical"]}
    first = next(iter(lines))
    for name, ev in by_address.items():
        if name not in named:   # vendor recognised only by its printed branch address (name is a logo, or unreadable)
            e = book.entries[name]
            cands.append({"value": e["name"], "lines": [], "pass": first, "vendor": e, "feats": {
                "canonical": 1.0, "from_address": 1.0, "addr_evidence": ev, "vendor_sim": 0.0}})
    vote("company", cands, len(lines))
    return cands


def ocr_addresses(lines: dict, company_by_pass: dict, tagger: LogReg) -> list[dict]:
    out = []
    for p, ls in lines.items():
        tagged = address_line_feats(ls, company_by_pass.get(p))
        for t in tagged:
            t["p"] = tagger.prob(t["feats"])
        a = _grow(tagged)
        if a:
            a["pass"] = p
            out.append(a)
    return out


def address_cands(lines: dict, ocr_addrs: list[dict], vendor: dict | None) -> list[dict]:
    """Candidates for the address chooser: each pass's OCR address, plus the matched vendor's known addresses."""
    n = len(lines)
    page = " ".join(l.text for ls in lines.values() for l in ls[:25])
    known = list(vendor["addresses"].values()) if vendor else []
    cands = []
    for a in ocr_addrs:
        votes = sum(norm(b["value"]) == norm(a["value"]) for b in ocr_addrs)
        cands.append({"value": a["value"], "pass": a["pass"], "lines": a["lines"], "feats": {
            "canonical": 0.0, "vendor_known": float(bool(vendor)), "n_known": min(len(known), 5) / 5,
            "evidence": max((sim(a["value"], k) for k in known), default=0.0),
            "votes": votes / n, "tagger": a["conf"], "length": min(len(a["value"]), 120) / 120}})
    texts = [a["value"] for a in ocr_addrs]
    evs = {k: VendorBook.address_evidence(k, texts, page) for k in known}
    for c in cands:
        f = c["feats"]
        f.update({"ev_exact": float(f["evidence"] >= 0.995), "votes_x_ev": f["votes"] * f["evidence"],
                  "ev_is_max": 0.0, "ev_margin": 0.0})
    for k in known:
        ev = evs[k]
        others = [v for kk, v in evs.items() if kk != k]
        support = max(ocr_addrs, key=lambda a: sim(k, a["value"]), default=None)
        cands.append({"value": k, "pass": support["pass"] if support else next(iter(lines)),
                      "lines": support["lines"] if support else [], "feats": {
                          "canonical": 1.0, "vendor_known": 1.0, "n_known": min(len(known), 5) / 5, "evidence": ev,
                          "votes": sum(sim(k, t) >= 0.8 for t in texts) / n,
                          "tagger": max((a["conf"] for a in ocr_addrs), default=0.0),
                          "length": min(len(k), 120) / 120,
                          # a vendor with several branches: does this branch beat the others on this receipt?
                          "ev_exact": float(ev >= 0.995), "ev_is_max": float(ev >= max(others, default=0.0)),
                          "ev_margin": max(min(ev - max(others, default=0.0), 0.2), -0.2) / 0.2,
                          "votes_x_ev": 0.0}})
    return cands


class Extractor:
    def __init__(self, path: Path = MODEL_PATH, book: VendorBook | None = None):
        spec = json.loads(Path(path).read_text())
        self.sets = {name: {k: LogReg(v) for k, v in models.items()} for name, models in spec["models"].items()}
        self.thresholds = spec.get("thresholds", {})
        self.book = book if book is not None else VendorBook.load(BOOK_PATH)

    def extract(self, doc: dict) -> dict:
        mode = "multi" if len(doc["passes"]) > 1 and "multi" in self.sets else "single"
        m = self.sets[mode]
        lines = ocr_lines(doc)
        out: dict = {}

        cc = company_cands(lines, self.book)
        for c in cc:
            c["p"] = m["company"].prob(c["feats"])
        ranked = sorted(cc, key=lambda c: -c["p"])
        out["company"] = _field("company", ranked, lines)
        best = ranked[0] if ranked else None
        vendor = best.get("vendor") if best else None
        out["company"]["known_vendor"] = bool(vendor)
        company_by_pass = {}
        for p in lines:
            bp = max((c for c in cc if c["pass"] == p and c["lines"]), key=lambda c: c["p"], default=None)
            company_by_pass[p] = bp["lines"][0] if bp else None

        for field, cands in date_total_candidates(lines).items():
            for c in cands:
                c["p"] = m[field].prob(c["feats"])
            out[field] = _field(field, sorted(cands, key=lambda c: -c["p"]), lines)

        ac = address_cands(lines, ocr_addresses(lines, company_by_pass, m["address_lines"]), vendor)
        for c in ac:
            c["p"] = m["address"].prob(c["feats"])
        out["address"] = _field("address", sorted(ac, key=lambda c: -c["p"]), lines)
        out["address"]["from_vendor_book"] = bool(ac) and max(ac, key=lambda c: c["p"])["feats"]["canonical"] == 1.0

        first = next(iter(lines))
        return {"mode": mode, "passes": list(lines), "fields": out,
                "lines": [{"text": l.text, "box": [l.x0, l.y0, l.x1, l.y1], "conf": round(l.conf, 1)} for l in lines[first]]}


def _box(ls, idxs):
    sel = [l for l in ls if l.idx in idxs]
    if not sel:
        return None
    return [min(l.x0 for l in sel), min(l.y0 for l in sel), max(l.x1 for l in sel), max(l.y1 for l in sel)]


def _field(field: str, ranked: list[dict], lines: dict) -> dict:
    if not ranked:
        return {"value": None, "conf": 0.0, "box": None, "alternatives": []}
    best = ranked[0]
    alts, seen = [], {key(field, best) if field in ("date", "total") else norm(best["value"])}
    for c in ranked[1:]:
        k = key(field, c) if field in ("date", "total") else norm(c["value"])
        if k not in seen:
            seen.add(k)
            alts.append({"value": c["value"], "conf": round(c["p"], 4)})
    f = {"value": best["value"], "conf": round(best["p"], 4), "pass": best["pass"],
         "box": _box(lines[best["pass"]], best["lines"]),
         "margin": round(best["p"] - (alts[0]["conf"] if alts else 0.0), 4), "alternatives": alts[:3]}
    if "raw" in best:
        f["raw"] = best["raw"]
    return f


def _grow(tagged: list[dict]) -> dict | None:
    """Take the strongest address line and grow it over neighbouring lines the tagger also accepts."""
    if not tagged:
        return None
    seed = max(range(len(tagged)), key=lambda i: tagged[i]["p"])
    if tagged[seed]["p"] < 0.5:
        return None
    lo = hi = seed
    while lo - 1 >= 0 and tagged[lo - 1]["p"] >= 0.5:
        lo -= 1
    while hi + 1 < len(tagged) and tagged[hi + 1]["p"] >= 0.5:
        hi += 1
    chosen = tagged[lo:hi + 1]
    text = clean(" ".join(t["text"] for t in chosen if t["text"]))
    if not text:
        return None
    return {"value": text, "conf": min(t["p"] for t in chosen), "lines": [t["line"].idx for t in chosen]}
