# Web UI + REST API in a container.
#   docker build -t swarm .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e TINYFISH_API_KEY -e SWARM_API_TOKEN=change-me swarm
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SWARM_STATE_DIR=/data/jobs SWARM_CACHE_DIR=/data/cache
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY swarm ./swarm
RUN pip install --no-cache-dir ".[server,pdf]" && useradd --create-home app && mkdir -p /data && chown app /data
USER app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["swarm", "serve", "--host", "0.0.0.0", "--port", "8000"]
