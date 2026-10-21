# Stage 1: Install dependencies and pre-train the development model.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# Install directly into the builder interpreter so training can import the dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY src/ ./src/
RUN PYTHONPATH=/app python -m src.train

# Stage 2: Runtime image. No compiler/build-essential packages are installed.
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app

# Copy the installed site-packages and executable entry points from the builder.
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
