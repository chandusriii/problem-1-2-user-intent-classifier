FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY tests/ ./tests/
COPY docs/ ./docs/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
RUN python src/train.py
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
