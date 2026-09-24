from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_meta_and_single_receipt():
    from api.extract import meta, preds
    m = meta()
    assert m["results"]["n_test"] == len(m["docs"]) == 200
    one = preds()[m["docs"][0]["id"]]
    assert set(one["fields"]) == {"company", "date", "address", "total"} and one["lines"]


def test_upload_path_runs_single_pass_extraction():
    from api.extract import upload
    doc = __import__("idp.ocrcache", fromlist=["get"]).get("000")
    words = doc["passes"]["fast6"]["words"]
    status, r = upload({"width": doc["width"], "height": doc["height"], "ms": 900, "words": words})
    assert status == 200 and r["mode"] == "single" and r["fields"]["total"]["value"] == "9.00"


def test_upload_rejects_bad_input():
    from api.extract import upload
    assert upload({})[0] == 400
    assert upload({"words": [{"t": "x"}]})[0] == 400
    assert upload({"words": [{"t": "a", "x": 0, "y": 0, "w": 1, "h": 1, "line": [1, 1, 1]}] * 4000})[0] == 413
