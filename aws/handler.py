"""AWS Lambda handler: S3 upload -> OCR -> extraction -> validation -> DynamoDB, with a review queue.

Triggered by s3:ObjectCreated on the inbox bucket. For each image:
  1. read it from S3 (the bucket is encrypted and private; this function can only read inbox/)
  2. run the IDP pipeline (4 Tesseract passes, field scorers, validation)
  3. write the result to DynamoDB, conditionally, so a retried or duplicated S3 event doesn't process twice
  4. if any field failed validation, send a message to the human-review SQS queue
  5. emit CloudWatch metrics in Embedded Metric Format (no extra API calls or permissions needed)
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

from idp.pipeline import process_image, record

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")
sqs = boto3.client("sqs")


def _decimal(x):
    if isinstance(x, float):
        return Decimal(str(x))
    if isinstance(x, dict):
        return {k: _decimal(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decimal(v) for v in x]
    return x


def _metrics(route: str, n_review_fields: int, ms: float):
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "IDP", "Dimensions": [["Route"]],
            "Metrics": [{"Name": "Documents", "Unit": "Count"}, {"Name": "FieldsToReview", "Unit": "Count"},
                        {"Name": "ProcessingMs", "Unit": "Milliseconds"}]}]},
        "Route": route, "Documents": 1, "FieldsToReview": n_review_fields, "ProcessingMs": ms}))


def process_record(rec: dict) -> dict:
    from PIL import Image

    bucket = rec["s3"]["bucket"]["name"]
    obj_key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
    doc_id = f"{obj_key}#{rec['s3']['object'].get('eTag', '')}"
    t = time.perf_counter()
    body = s3.get_object(Bucket=bucket, Key=obj_key)["Body"].read()
    result = process_image(Image.open(io.BytesIO(body)), doc_id)
    row = record(doc_id, result)
    row["source"] = f"s3://{bucket}/{obj_key}"
    row["processed_at"] = int(time.time())
    table = ddb.Table(os.environ["RESULTS_TABLE"])
    try:
        table.put_item(Item=_decimal(row), ConditionExpression="attribute_not_exists(doc_id)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return {"doc_id": doc_id, "status": "duplicate"}
        raise
    if row["route"] == "review":
        sqs.send_message(QueueUrl=os.environ["REVIEW_QUEUE_URL"], MessageBody=json.dumps({
            "doc_id": doc_id, "source": row["source"], "review_fields": row["review_fields"],
            "fields": {k: row["fields"][k] for k in row["review_fields"]}}, default=str))
    ms = round((time.perf_counter() - t) * 1000, 1)
    _metrics(row["route"], len(row["review_fields"]), ms)
    return {"doc_id": doc_id, "status": "processed", "route": row["route"]}


def handler(event, context=None):
    return [process_record(r) for r in event.get("Records", []) if r.get("eventSource") == "aws:s3"]
