FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/models", "/data", "/tmp/llama-logs"]

EXPOSE 5000
EXPOSE 42069
EXPOSE 8000-8020

ENV MODELS_DIR=/models
ENV DATA_DIR=/data
ENV LOGS_DIR=/tmp/llama-logs
ENV PORT_RANGE_START=8000
ENV PORT_RANGE_END=8020
ENV INTERNAL_PORT_RANGE_START=9000
ENV INTERNAL_PORT_RANGE_END=9020
ENV LLAMAMAN_MAX_MODELS=1
ENV LLAMAMAN_PROXY_PORT=42069
ENV LLAMAMAN_IDLE_TIMEOUT=0
ENV LLAMAMAN_PDF_MAX_CONCURRENT=4
ENV LLAMA_NETWORK=llamaman-net
ENV LLAMA_CONTAINER_PREFIX=llamaman-

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT []
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
