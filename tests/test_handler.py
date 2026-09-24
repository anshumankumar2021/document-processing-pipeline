"""The Lambda handler end to end against mocked S3, DynamoDB and SQS (moto)."""
import io
import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="inbox")
        ddb = boto3.client("dynamodb")
        ddb.create_table(TableName="results", BillingMode="PAY_PER_REQUEST",
                         AttributeDefinitions=[{"AttributeName": "doc_id", "AttributeType": "S"}],
                         KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}])
        q = boto3.client("sqs").create_queue(QueueName="review")["QueueUrl"]
        monkeypatch.setenv("RESULTS_TABLE", "results")
        monkeypatch.setenv("REVIEW_QUEUE_URL", q)
        import importlib
        import sys
        sys.path.insert(0, str(ROOT / "aws"))
        import handler
        importlib.reload(handler)   # create clients inside the mock
        yield s3, q, handler


def upload(s3, key, doc_id, monkeypatch, handler):
    # skip Tesseract: return the cached OCR for this receipt
    doc = __import__("idp.ocrcache", fromlist=["get"]).get(doc_id)
    monkeypatch.setattr("idp.ocr.run_ocr", lambda image, doc_id="upload", passes=None: doc)
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (10, 10), 255).save(buf, "PNG")
    s3.put_object(Bucket="inbox", Key=key, Body=buf.getvalue())
    etag = s3.head_object(Bucket="inbox", Key=key)["ETag"]
    return {"Records": [{"eventSource": "aws:s3", "s3": {"bucket": {"name": "inbox"}, "object": {"key": key, "eTag": etag}}}]}


def review_doc():
    preds = json.loads((ROOT / "results" / "test_predictions.json").read_text())
    return next(i for i, p in sorted(preds.items()) if p["route"] == "review")


def test_processes_upload_writes_result_and_queues_review(aws, monkeypatch, capsys):
    s3, q, handler = aws
    doc = review_doc()
    ev = upload(s3, f"inbox/{doc}.jpg", doc, monkeypatch, handler)
    out = handler.handler(ev)
    assert out[0]["status"] == "processed" and out[0]["route"] == "review"
    item = boto3.resource("dynamodb").Table("results").get_item(Key={"doc_id": out[0]["doc_id"]})["Item"]
    assert item["route"] == "review" and set(item["fields"]) == {"company", "date", "address", "total"}
    msgs = boto3.client("sqs").receive_message(QueueUrl=q, MaxNumberOfMessages=10).get("Messages", [])
    assert len(msgs) == 1 and json.loads(msgs[0]["Body"])["review_fields"] == item["review_fields"]
    emf = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert emf and emf[-1]["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "IDP"


def test_duplicate_event_is_not_processed_twice(aws, monkeypatch):
    s3, q, handler = aws
    doc = review_doc()
    ev = upload(s3, f"inbox/{doc}.jpg", doc, monkeypatch, handler)
    handler.handler(ev)
    again = handler.handler(ev)
    assert again[0]["status"] == "duplicate"
    msgs = boto3.client("sqs").receive_message(QueueUrl=q, MaxNumberOfMessages=10).get("Messages", [])
    assert len(msgs) == 1
