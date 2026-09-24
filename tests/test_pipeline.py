import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = "000"   # a cached receipt (OCR in data/ocr.jsonl.gz), so these tests don't need Tesseract


def cached(doc_id=DOC):
    return __import__("idp.ocrcache", fromlist=["get"]).get(doc_id)


def test_pipeline_on_cached_ocr():
    from idp.pipeline import process_ocr, record
    r = process_ocr(cached())
    assert r["mode"] == "multi" and set(r["fields"]) == {"company", "date", "address", "total"}
    assert r["fields"]["total"]["value"] == "9.00" and r["fields"]["date"]["value"] == "2018-12-25"
    assert r["route"] in ("auto", "review") and all(c["ok"] or c["reasons"] for c in r["checks"].values())
    row = record("x", r)
    assert set(row) >= {"doc_id", "route", "fields", "review_fields"} and "lines" not in row


def test_single_pass_uses_single_models():
    from idp.pipeline import process_ocr
    d = cached()
    r = process_ocr({**d, "passes": {"fast6": d["passes"]["fast6"]}})
    assert r["mode"] == "single"


def test_inference_needs_no_numpy_or_sklearn():
    # the Lambda image and the Vercel function ship without numpy/scikit-learn
    import subprocess, sys
    code = ("import sys; sys.modules['numpy']=None; sys.modules['sklearn']=None; "
            "import json; from idp.pipeline import process_ocr; "
            f"from idp.ocrcache import get; print(process_ocr(get('{DOC}'))['route'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.skipif(not os.path.exists("/usr/bin/tesseract"), reason="needs the tesseract binary")
def test_real_ocr_on_a_rendered_receipt():
    from PIL import Image, ImageDraw, ImageFont
    from idp.pipeline import process_image
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        pytest.skip("no TrueType font available")
    im = Image.new("L", (520, 520), 255)
    d = ImageDraw.Draw(im)
    for k, t in enumerate(["KEDAI MAJU TRADING SDN BHD", "(123456-X)", "NO 12, JALAN SAGU 18,", "81100 JOHOR BAHRU, JOHOR.",
                           "Date: 25/12/2018  8:13 PM", "ITEM A          5.00", "ITEM B          4.00", "TOTAL           9.00",
                           "CASH           10.00", "CHANGE          1.00"]):
        d.text((20, 20 + k * 48), t, fill=0, font=font)
    r = process_image(im)
    assert r["fields"]["total"]["value"] == "9.00"
    assert r["fields"]["date"]["value"] == "2018-12-25"
