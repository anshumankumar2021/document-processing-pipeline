"""Download the SROIE 2019 receipts (corrected labels from github.com/zzzDavid/ICDAR-2019-SROIE, MIT licence)
and Tesseract's tessdata_best English model, then write data/labels.json and a fixed train/test split.

    python -m scripts.prepare_data
"""
from __future__ import annotations

import json
import random
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/zzzDavid/ICDAR-2019-SROIE"
COMMIT = "27be4271b251c256f695acbade9a801bffe85994"
SRC = ROOT / "data" / "sroie"
N_TEST = 200
TESSDATA_BEST = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata"


def main():
    if not (SRC / "data" / "key").exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPO, str(SRC)], check=True)
        subprocess.run(["git", "-C", str(SRC), "checkout", "--quiet", COMMIT], check=True)
    best = ROOT / "data" / "tessdata" / "eng.traineddata"
    if not best.exists():
        best.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TESSDATA_BEST, best)
    labels = {}
    for f in sorted((SRC / "data" / "key").glob("*.json")):
        k = json.loads(f.read_text())
        labels[f.stem] = {fld: k.get(fld, "") for fld in ("company", "date", "address", "total")}
    ids = sorted(labels)
    rng = random.Random(2019)
    test = sorted(rng.sample(ids, N_TEST))
    split = {"train": [i for i in ids if i not in set(test)], "test": test}
    (ROOT / "data" / "labels.json").write_text(json.dumps(labels, indent=0, ensure_ascii=False))
    (ROOT / "data" / "split.json").write_text(json.dumps(split))
    print(f"{len(labels)} receipts: {len(split['train'])} train, {len(split['test'])} test")


if __name__ == "__main__":
    main()
