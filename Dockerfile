# Multi-Agent Swarm: web UI, REST API and job workers.
#
#   docker build -t swarm .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e TINYFISH_API_KEY -e SWARM_API_TOKEN=change-me \
#       -v swarm-data:/data swarm
#
# The same image runs every role: `swarm serve` (default: API + one embedded worker),
# `swarm serve --no-worker` (API only) and `swarm worker` (jobs only). See deploy/ for
# Docker Compose with HTTPS, Kubernetes and systemd.

# ---- build: a wheel from the source tree (no compilers or source in the final image)
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY swarm ./swarm
RUN pip install --no-cache-dir build==1.2.2 && python -m build --wheel --outdir /dist

# ---- runtime
FROM python:3.12-slim
LABEL org.opencontainers.image.title="multi-agent-swarm" \
      org.opencontainers.image.source="https://github.com/Chetan0246/multi-agent-swarm" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SWARM_STATE_DIR=/data/state SWARM_CACHE_DIR=/data/cache \
    SWARM_LOG_FORMAT=json HOME=/home/app

COPY --from=build /dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/*.whl) && pip install "${WHEEL}[server,pdf]" && rm -f /tmp/*.whl \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /data && chown app:app /data

USER 10001:10001
WORKDIR /home/app
VOLUME /data
EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["swarm", "serve", "--host", "0.0.0.0", "--port", "8000"]
