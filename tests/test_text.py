import datetime as dt

from idp.score import correct
from idp.text import find_dates, find_money, parse_date, parse_money
from idp.validate import validate


def test_dates_in_receipt_formats():
    assert parse_date("25/12/2018") == dt.date(2018, 12, 25)
    assert parse_date("Date: 02-01-18 13:05") == dt.date(2018, 1, 2)
    assert parse_date("12 MAR 2018") == dt.date(2018, 3, 12)
    assert parse_date("2018-03-12") == dt.date(2018, 3, 12)
    assert parse_date("03/25/2018") == dt.date(2018, 3, 25)          # month-first when day-first is impossible
    assert parse_date("31/02/2018") is None                           # not a real date
    assert [d["raw"] for d in find_dates("Tel 012-345 6789")] == []   # phone numbers aren't dates


def test_money():
    assert [m["value"] for m in find_money("TOTAL RM 1,234.50  CASH 9,00")] == [1234.5, 9.0]
    assert parse_money("RM 9.00") == 9.0 and parse_money("$12.30") == 12.3 and parse_money("45") == 45.0
    assert find_money("Tel 012-3456789") == []


def test_scoring_rules():
    assert correct("total", "9.00", "RM9.00")
    assert correct("date", "2018-12-25", "25/12/2018")
    assert correct("company", "book ta .k  (taman daya) sdn bhd", "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert not correct("company", "BOOK TA -K (TAMAN DAYA) SDN BHD", "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert correct("company", "BOOK TA -K (TAMAN DAYA) SDN BHD", "BOOK TA .K (TAMAN DAYA) SDN BHD", near=True)


def test_validation_routes_uncertain_or_invalid_fields():
    th = {"company": 0.5, "date": 0.5, "address": 0.5, "total": 0.5}
    good = {"company": {"value": "ABC TRADING", "conf": 0.9}, "date": {"value": "2018-05-01", "conf": 0.9},
            "address": {"value": "12 JALAN SAGU, 81100 JOHOR", "conf": 0.9}, "total": {"value": "9.00", "conf": 0.9}}
    assert validate(good, th, today=dt.date(2026, 1, 1))["route"] == "auto"
    bad = {**good, "date": {"value": "2031-05-01", "conf": 0.99}, "total": {"value": "9.00", "conf": 0.2}}
    v = validate(bad, th, today=dt.date(2026, 1, 1))
    assert v["route"] == "review" and set(v["review_fields"]) == {"date", "total"}
    assert "future" in v["checks"]["date"]["reasons"][0] and "low confidence" in v["checks"]["total"]["reasons"][0]
    assert validate({**good, "address": {"value": "JALAN SAGU", "conf": 0.9}}, th)["checks"]["address"]["reasons"]
