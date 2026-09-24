# Lambda container image: Debian + Tesseract 5 + the IDP package, run through the Lambda runtime interface client.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng curl ca-certificates \
 && mkdir -p /opt/tessdata-best \
 && curl -sSfL -o /opt/tessdata-best/eng.traineddata \
      https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task
COPY aws/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY idp ./idp
COPY models ./models
COPY aws/handler.py ./handler.py

ENV TESSDATA_BEST=/opt/tessdata-best OMP_THREAD_LIMIT=1
# no root at runtime
RUN useradd --uid 10001 --no-create-home lambda
USER 10001
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["handler.handler"]
