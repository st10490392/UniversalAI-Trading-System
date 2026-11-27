FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py three_setup_portfolio.json ./
COPY config/ ./config/
RUN useradd --create-home app && chown app:app /app
USER app
CMD ["sh","-c","exec python -m uvicorn paper_control:app --host 0.0.0.0 --port ${PORT:-8000}"]
