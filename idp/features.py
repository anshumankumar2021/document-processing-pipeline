"""Candidate generation and features for each field.

Every field is extracted the same way: generate candidates from the OCR lines, describe each with a few dozen
features, and let a logistic-regression scorer (trained in scripts/train.py) pick the best one. The winning
candidate's probability is the field's confidence.
"""
from __future__ import annotations

import re
import statistics

from idp.ocr import Line
from idp.text import find_dates, find_money

COMPANY_KW = re.compile(r"\b(SDN|BHD|BERHAD|ENTERPRISE|TRADING|RESTAURANT|RESTORAN|STORE|MART|SUPERMARKET|HARDWARE|"
                        r"CO\b|COMPANY|PLT|MARKETING|BAKERY|CAFE|FOOD|STATIONERY|BOOK|SHOP|CENTRE|CENTER|INDUSTRIES|"
                        r"HOLDINGS|TRADERS?|SERVICES?|GIFT|MACHINERY|MOTOR|PHARMACY|KEDAI|\(M\))", re.I)
REG_NO = re.compile(r"(\(?\d{5,7}\s?-\s?[A-Z]\)?|CO\.?\s?REG|COMPANY\s?NO|ROC|GST\s?(REG|ID|NO))", re.I)
ADDRESS_KW = re.compile(r"\b(JALAN|JLN|TAMAN|TMN|LOT|NO\b|NO\.|BLOK|BLOCK|LORONG|KAMPUNG|KG|BANDAR|PERSIARAN|BATU|"
                        r"ROAD|STREET|AVENUE|PUSAT|INDUSTRI|PERINDUSTRIAN|KAWASAN|JUSCO|MALL|PLAZA|WISMA|SEKSYEN|"
                        r"SECTION|SS\d|PJU|USJ|BUKIT|SUNGAI|SG\b|GROUND|FLOOR|TINGKAT)", re.I)
STATES = re.compile(r"\b(JOHOR|SELANGOR|KUALA\s?LUMPUR|PENANG|PULAU\s?PINANG|PERAK|KEDAH|MELAKA|MALACCA|NEGERI|"
                    r"SEMBILAN|PAHANG|KELANTAN|TERENGGANU|SABAH|SARAWAK|PUTRAJAYA|W\.?P\.?|MALAYSIA|SHAH\s?ALAM|"
                    r"PETALING|KLANG|SERI\s?KEMBANGAN|CHERAS|PUCHONG|KAJANG|SUBANG)\b", re.I)
POSTCODE = re.compile(r"(?<!\d)\d{5}(?!\d)")
CONTACT = re.compile(r"\b(TEL|PHONE|FAX|H/?P|EMAIL|E-MAIL|WWW|HTTP|GST|SST|TAX\s?INV|INVOICE|RECEIPT|DATE|CASHIER|"
                     r"TABLE|DOC(UMENT)?)\b", re.I)
TOTAL_KW = re.compile(r"(T[O0]TAL|AMOUNT|AMT|NETT?|GRAND|DUE|PAYABLE|ROUND(ED|ING)?|INCL)", re.I)
TOTAL_WORD = re.compile(r"T[O0]TAL", re.I)
ROUND_KW = re.compile(r"ROUND", re.I)
PAY_KW = re.compile(r"(CASH|CHANGE|TENDER|RECEIVED|CREDIT|VISA|MASTER|CARD|DEBIT|BALANCE)", re.I)
PAYABLE_KW = re.compile(r"(PAYABLE|TO\s?BE\s?PAID|AMOUNT\s?DUE|NETT?\s?T[O0]TAL|GRAND|INCL(USIVE)?)", re.I)
TENDER_KW = re.compile(r"(CASH|TENDER|RECEIVED|PAID\s?BY)", re.I)
CHANGE_KW = re.compile(r"(CHANGE|BALANCE)", re.I)
SUBTOTAL_KW = re.compile(r"(SUB\s?-?\s?T[O0]TAL|EXCL|EXCLUDING|BEFORE)", re.I)
TAX_KW = re.compile(r"(GST|SST|TAX|SERVICE)", re.I)
NOT_TOTAL_KW = re.compile(r"(GST|SST|TAX|DISC|QTY|SUB\s?-?T[O0]TAL|ITEM|SAVING|PRICE|UNIT|EXCL|SERVICE\s?CHARGE)", re.I)
DATE_KW = re.compile(r"(DATE|TARIKH|DT\b)", re.I)
TIME = re.compile(r"\d{1,2}:\d{2}")


def clean(s: str) -> str:
    """Strip OCR junk from the ends of a line, keeping '.', ',' and ')' which are part of names and addresses."""
    return s.strip(" :;|_-—=*~'\"`«»“”‘’").strip()


def _ratio(pattern, s: str) -> float:
    return sum(1 for ch in s if pattern(ch)) / max(1, len(s))


def line_feats(l: Line, lines: list[Line], H: float, med_h: float) -> dict:
    t = l.text
    letters = [c for c in t if c.isalpha()]
    return {
        "y_rel": l.y0 / H, "idx": min(l.idx, 30) / 30, "n_chars": min(len(t), 60) / 60,
        "alpha": _ratio(str.isalpha, t), "digit": _ratio(str.isdigit, t),
        "upper": sum(c.isupper() for c in letters) / max(1, len(letters)),
        "conf": l.conf / 100, "height": l.height / med_h if med_h else 1.0,
        "commas": min(t.count(","), 5) / 5, "colon": float(":" in t),
    }


def context(lines: list[Line]) -> tuple[float, float]:
    H = max((l.y1 for l in lines), default=1) or 1
    med_h = statistics.median([l.height for l in lines]) if lines else 1
    return H, med_h


# ---------------- company ----------------
def company_candidates(lines: list[Line]) -> list[dict]:
    H, med_h = context(lines)
    cands = []
    top = [l for l in lines if l.idx < 12]
    first_alpha = next((l.idx for l in top if sum(c.isalpha() for c in l.text) >= 4), -1)
    reg_idx = next((l.idx for l in top if REG_NO.search(l.text)), 99)
    for l in top:
        t = clean(l.text)
        if sum(c.isalpha() for c in t) < 3:
            continue
        f = line_feats(l, lines, H, med_h)
        nxt = lines[l.idx + 1].text if l.idx + 1 < len(lines) else ""
        f.update({
            "kw_company": float(bool(COMPANY_KW.search(t))), "kw_sdn_bhd": float(bool(re.search(r"SDN|BHD", t, re.I))),
            "next_is_reg": float(bool(REG_NO.search(nxt))), "is_reg": float(bool(REG_NO.search(t))),
            "before_reg": float(l.idx < reg_idx), "dist_to_reg": min(abs(reg_idx - l.idx), 10) / 10,
            "is_first_alpha": float(l.idx == first_alpha), "rank_from_first": min(max(l.idx - first_alpha, 0), 8) / 8,
            "kw_address": float(bool(ADDRESS_KW.search(t) or POSTCODE.search(t))), "kw_contact": float(bool(CONTACT.search(t))),
            "has_digit": float(any(c.isdigit() for c in t)), "lower_start": float(t[:1].islower()),
        })
        f["two_lines"] = 0.0
        cands.append({"value": t, "lines": [l.idx], "feats": f})
    # names wrapped over two lines ("POPULAR BOOK" / "CO. (M) SDN BHD")
    singles = {c["lines"][0]: c for c in cands}
    for i, c in list(singles.items()):
        nxt = singles.get(i + 1)
        if nxt and i < 8 and not c["feats"]["kw_address"] and not nxt["feats"]["kw_address"] and not nxt["feats"]["is_reg"]:
            f = dict(c["feats"])
            f.update({"two_lines": 1.0, "kw_company": max(c["feats"]["kw_company"], nxt["feats"]["kw_company"]),
                      "kw_sdn_bhd": max(c["feats"]["kw_sdn_bhd"], nxt["feats"]["kw_sdn_bhd"]),
                      "next_is_reg": nxt["feats"]["next_is_reg"], "n_chars": min(c["feats"]["n_chars"] + nxt["feats"]["n_chars"], 1.0)})
            cands.append({"value": f"{c['value']} {nxt['value']}", "lines": [i, i + 1], "feats": f})
    return cands


# ---------------- address ----------------
def address_line_feats(lines: list[Line], company_idx: int | None) -> list[dict]:
    """Per-line features for the address tagger. Lines tagged as address are joined in order."""
    H, med_h = context(lines)
    stop = next((l.idx for l in lines if find_dates(l.text) or re.search(r"INVOICE|RECEIPT|CASHIER", l.text, re.I)), len(lines))
    out = []
    for l in lines[:25]:
        t = clean(l.text)
        f = line_feats(l, lines, H, med_h)
        f.update({
            "kw_address": min(len(ADDRESS_KW.findall(t)), 3) / 3, "state": float(bool(STATES.search(t))),
            "postcode": float(bool(POSTCODE.search(t))), "kw_contact": float(bool(CONTACT.search(t))),
            "kw_company": float(bool(COMPANY_KW.search(t))), "is_reg": float(bool(REG_NO.search(t))),
            "after_company": float(company_idx is not None and l.idx > company_idx),
            "is_company": float(company_idx == l.idx),
            "dist_company": min(abs(l.idx - company_idx), 10) / 10 if company_idx is not None else 1.0,
            "before_stop": float(l.idx < stop), "ends_comma": float(t.endswith(",")),
            "short": float(len(t) < 4),
        })
        out.append({"line": l, "text": t, "feats": f})
    return out


# ---------------- date ----------------
def date_candidates(lines: list[Line]) -> list[dict]:
    H, med_h = context(lines)
    found = []
    for l in lines:
        for d in find_dates(l.text):
            found.append((l, d))
    counts: dict = {}
    for _, d in found:
        if d["date"]:
            counts[d["date"]] = counts.get(d["date"], 0) + 1
    cands = []
    for rank, (l, d) in enumerate(found):
        f = line_feats(l, lines, H, med_h)
        f.update({
            "kw_date": float(bool(DATE_KW.search(l.text))), "has_time": float(bool(TIME.search(l.text))),
            "parsed": float(d["date"] is not None), "short_year": float(d["short_year"]),
            "k_dmy": float(d["kind"] == "dmy"), "k_ymd": float(d["kind"] == "ymd"), "k_month": float(d["kind"] == "dMy"),
            "repeats": min(counts.get(d["date"], 0), 3) / 3, "first": float(rank == 0),
            "plausible_year": float(bool(d["date"]) and 2000 <= d["date"].year <= 2030),
            "n_cands": min(len(found), 5) / 5,
        })
        cands.append({"value": d["date"].isoformat() if d["date"] else d["raw"], "raw": d["raw"], "lines": [l.idx],
                      "feats": f, "date": d["date"]})
    return cands


# ---------------- total ----------------
def total_candidates(lines: list[Line]) -> list[dict]:
    H, med_h = context(lines)
    found = []
    for l in lines:
        for m in find_money(l.text):
            found.append((l, m))
    if not found:
        return []
    vals = [m["value"] for _, m in found]
    vmax = max(vals) or 1.0
    pay_vals = [m["value"] for l, m in found if PAY_KW.search(l.text)]
    non_pay = [m["value"] for l, m in found if not PAY_KW.search(l.text)] or [0]
    counts: dict = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    order = sorted(set(vals), reverse=True)
    # arithmetic that pins the total down: cash - change, and subtotal + tax (+ rounding)
    def amounts(pat, exclude=None):
        return {m["value"] for l, m in found if pat.search(l.text) and not (exclude and exclude.search(l.text))}
    tender = amounts(TENDER_KW, CHANGE_KW)
    change = amounts(CHANGE_KW)
    implied_by_payment = {round(t - c, 2) for t in tender for c in change if t > c}
    subs, taxes = amounts(SUBTOTAL_KW), amounts(TAX_KW, SUBTOTAL_KW)
    rounds = amounts(ROUND_KW)
    implied_by_sum = {round(a + t + sgn * r, 2) for a in subs for t in taxes | {0.0} for r in rounds | {0.0} for sgn in (1, -1)
                      if t or r}
    cands = []
    for l, m in found:
        v = m["value"]
        t = l.text
        right = t[m["end"]:].strip()
        f = line_feats(l, lines, H, med_h)
        f.update({
            "kw_total": float(bool(TOTAL_KW.search(t))), "kw_total_word": float(bool(TOTAL_WORD.search(t))),
            "kw_round": float(bool(ROUND_KW.search(t))), "kw_pay": float(bool(PAY_KW.search(t))),
            "kw_not_total": float(bool(NOT_TOTAL_KW.search(t))), "has_rm": float("RM" in t.upper() or "$" in t),
            "is_max": float(v == vmax), "is_max_non_pay": float(v == max(non_pay)), "rel_value": v / vmax,
            "rank": min(order.index(v), 5) / 5, "repeats": min(counts[v], 4) / 4,
            "equals_payment": float(v in pay_vals), "rightmost": float(not any(ch.isdigit() for ch in right)),
            "zero": float(v == 0), "y_below_half": float(l.y0 / H > 0.5),
            "prev_kw_total": float(l.idx > 0 and bool(TOTAL_WORD.search(lines[l.idx - 1].text))),
            "kw_payable": float(bool(PAYABLE_KW.search(t))), "kw_subtotal": float(bool(SUBTOTAL_KW.search(t))),
            "cash_minus_change": float(any(abs(v - x) < 0.005 for x in implied_by_payment)),
            "sum_of_parts": float(any(abs(v - x) < 0.005 for x in implied_by_sum)),
        })
        cands.append({"value": f"{v:.2f}", "num": v, "raw": m["raw"], "lines": [l.idx], "feats": f})
    return cands
