# Intelligent Document Processing: receipts

[![ci](https://github.com/anshumankumar2021/document-processing-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/anshumankumar2021/document-processing-pipeline/actions/workflows/ci.yml)

**Live demo: [document-processing-pipeline-theta.vercel.app](https://document-processing-pipeline-theta.vercel.app)**

A pipeline that turns real scanned receipts into structured records (company, date, address, total) and sends
anything it isn't sure about to a person instead of the database. It is built around one question: *when is
the reading wrong?* Four OCR passes vote on each field, small learned models pick the best candidate and report
a calibrated confidence, and validation routes low-confidence or invalid fields to a human-review queue.

It runs as an event-driven serverless pipeline on AWS (S3 → container Lambda → DynamoDB / SQS review queue),
defined as an AWS SAM template and tested end to end against mocked AWS services in CI.

```
 S3 inbox/ ──► Lambda (container) ─────────────────────────────────────────────► DynamoDB  (auto-accepted)
 (KMS, TLS)    4 Tesseract passes ─► field scorers + OCR voting ─► validation ─┤
                                     + vendor matching                         └► SQS review queue  (+ reasons)
```

## Data

[SROIE 2019](https://rrc.cvc.uab.es/?ch=13) (ICDAR): 626 real scanned receipts from Malaysian shops with labelled
company, date, address and total, using the corrected labels from
[zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) (MIT). Split with a fixed seed into
**426 training and 200 test receipts**. The test receipts are used only by `scripts/evaluate.py`.

Scoring is strict. Amounts and dates must match by value. Company and address must match the label exactly after
upper-casing and collapsing whitespace. A near-match rate (edit similarity ≥ 0.9) is reported alongside.

## Results (200 unseen receipts)

| Field | OCR ceiling¹ | 1 OCR pass | **4 passes + voting** | Near match | Auto-accepted | Auto precision |
|---|---|---|---|---|---|---|
| Total | 93.0% | 78.5% | **85.5%** | – | 76% | 95.4% |
| Date | 92.5% | 80.5% | **87.0%** | – | 80% | 95.6% |
| Company | 55.5% | 67.0% | **72.5%** | 81.0% | 50% | 94.0% |
| Address | 42.0% | 48.0% | **47.5%** | 80.0% | 15% | 93.3% |

¹ How often *any* OCR pass produced the exact right value. Company and address can beat it because a
recognised vendor's name and address come from the vendor book (below), not from the OCR text.

**Review routing.** Thresholds are chosen by 5-fold cross-validation on the training set to target 95% precision.
On the test set:

- **55.2%** of field values are auto-accepted, and **95.0%** of those are correct.
- Wrong values written to the database: **215 → 22 (−89.8%)**, compared with writing every extracted value.
  Fields sent to review are assumed to be corrected by the reviewer.
- **10.5%** of receipts go straight through with no human touch; 90.5% of those are entirely correct.
- Vendors seen in training (126 test receipts): company **92.9%** exact. New vendors (74): 37.8%, straight from
  OCR. That gap is why new vendors go to review.

**What each piece bought** (test set, exact match):

| Change | Effect |
|---|---|
| 4 OCR passes with voting instead of 1 | total +7.0 pts, date +6.5, company +5.5; OCR time p50 0.6 s → 4.1 s on one core |
| Vendor book (known names and branch addresses) | company for known vendors 92.9%; address evidence lets logo-only receipts be recognised |
| Arithmetic cross-checks (cash − change, subtotal + tax + rounding) | total 79.5% → 85.5% |
| Calibrated thresholds + schema rules | 89.8% fewer wrong values stored at 55% automation |

Extraction and validation take 34 ms p50 per receipt; OCR dominates (4.1 s p50 for four passes, 9.7 s p95).

### Honest limits

- The data is 626 receipts from one country, and the test set is only 200 receipts, so per-field numbers move
  by about ±3 points between splits.
- Address exact match is hard: labels are multi-line and punctuation varies between receipts of the same branch.
  Near match is 80%, but only 15% of addresses clear the auto-accept bar.
- "Errors reaching the database" assumes reviewers fix everything they are sent. The review workload (45% of
  fields) is the cost of that.
- OCR is Tesseract. A stronger OCR model would raise every ceiling. The extractor and routing don't depend on it.

## How it works

1. **OCR** (`idp/ocr.py`): four Tesseract 5 passes: the standard and `tessdata_best` English models, each with
   page-segmentation modes 6 and 4. RapidOCR was tried first; its default model drops the spaces between words,
   which makes exact matching of names and addresses impossible. Upscaling the images made Tesseract worse.
2. **Candidates and features** (`idp/features.py`): dates and amounts from OCR-tolerant patterns; company from
   the top lines (including names wrapped over two lines); address lines from a tagger. Each candidate has
   position, font-size, keyword, OCR-confidence and arithmetic features, plus how many passes agree on it.
3. **Vendor book** (`idp/vendors.py`): names and branch addresses from the *training* labels. An OCR line close
   to a known vendor yields that vendor's canonical name. A known branch address printed on the receipt
   identifies the vendor even when the name is a logo. While training, each receipt is left out of its own book.
4. **Scorers** (`idp/extract.py`, `scripts/train.py`): one logistic regression per field, trained with
   scikit-learn and exported as JSON weights. Inference needs only the standard library plus rapidfuzz.
   There are separate model sets for 4-pass and single-pass OCR.
5. **Validation** (`idp/validate.py`): schema rules (a plausible, non-future date; a positive amount; a name
   that looks like a name; an address with a number or postcode) plus the learned confidence thresholds.

## AWS deployment (`aws/`)

- `template.yaml` (AWS SAM):
  - **S3 inbox**: KMS encryption, all public access blocked, TLS-only bucket policy, versioning, 90-day lifecycle.
  - **Container Lambda**: triggered on `inbox/` uploads; reserved concurrency 20; failed events go to a
    dead-letter queue.
  - **DynamoDB results table**: encrypted, point-in-time recovery.
  - **SQS review queue**: encrypted.
  - **CloudWatch alarms**: function errors, failed events, and review backlog age.
- **Least-privilege IAM**: the function can `GetObject` only on `inbox/*`, `PutItem` on one table, and
  `SendMessage` to its two queues.
- `handler.py`:
  - **Idempotent**: a conditional write keyed on object key + ETag, so a retried or duplicated S3 event is
    processed once.
  - **Metrics** in CloudWatch Embedded Metric Format, with no extra API calls.
- `Dockerfile`: Debian + Tesseract 5 + `tessdata_best`, run as a non-root user through the Lambda runtime
  interface client.

```bash
sam build && sam deploy --guided
aws s3 cp receipt.jpg s3://idp-inbox-<account>-<region>/inbox/
```

CI (below) builds this image and processes a real receipt inside it. The handler is tested against moto.

## Live demo

The [demo](https://document-processing-pipeline-theta.vercel.app) has two tabs:
- **Browse**: all 200 test receipts, each showing the image, boxes around the extracted fields, confidence
  against the review threshold, the routing decision with reasons, and the label.
- **Upload your own**: runs Tesseract.js in the browser (the same engine compiled to WebAssembly). Only the
  recognised words are sent to the serverless extractor (`api/extract.py`), which uses the single-pass models.
  Nothing is stored.

Vercel functions can't run the Tesseract binary, which is why OCR happens in the browser.

## Run it

```bash
pip install -r requirements-dev.txt        # plus the tesseract binary for OCR (apt install tesseract-ocr)
pytest -q                                  # parsing, validation, pipeline, Lambda handler (moto), API
python -m scripts.prepare_data             # download SROIE + tessdata_best, labels, fixed split
python -m scripts.run_ocr                  # 4 OCR passes per receipt -> data/ocr.jsonl.gz (cached in the repo)
python -m scripts.train                    # field scorers, vendor book, CV thresholds -> models/
python -m scripts.evaluate                 # test-set results -> results/
python -m scripts.dev_server               # demo at http://localhost:3004
```

CI runs three jobs:
- **Tests**: unit tests plus `cfn-lint` on the template.
- **Reproduce**: downloads the dataset, re-OCRs a sample and requires it to match the cache word for word,
  retrains, re-scores, and fails if `models/` or `results/` change.
- **Lambda image**: builds the image and processes a real receipt inside it.

## Layout

```
idp/          OCR passes, candidates + features, extraction, vendor book, validation, pipeline, scoring
scripts/      prepare_data, run_ocr, train, evaluate, dev_server
aws/          SAM template, Lambda handler, Dockerfile
api/          Vercel serverless endpoint for the demo
public/       demo page
models/       trained weights (fields.json) and vendor book (vendors.json)
results/      results.json and per-receipt test predictions
data/         labels.json, split.json, ocr.jsonl.gz (OCR cache)
tests/
```
