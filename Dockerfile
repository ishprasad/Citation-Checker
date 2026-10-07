# Citation Checker: FastAPI app + static UI in one container.
#
#   docker build -t citation-checker .
#   docker run --env-file .env -p 8000:8000 -v citation-cache:/app/.cache citation-checker
#
# The volume keeps downloads, model-call cache and finished checks across restarts.
# Optional GROBID for better PDF parsing: run lfoppiano/grobid and set GROBID_URL (see README).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY demo ./demo
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 checker \
    && mkdir -p /app/.cache \
    && chown -R checker:checker /app
USER checker

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# One worker: jobs, rate limits and live progress events live in this process.
# Behind a reverse proxy, set TRUST_PROXY_HEADERS=true so rate limits use the real client IP.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
