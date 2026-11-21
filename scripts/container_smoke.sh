#!/usr/bin/env bash
# Smoke-test the container image the way production runs it: read-only filesystem, no
# capabilities, non-root, token auth. Submits an offline (mock) research job over HTTP and
# checks health, auth, the job result, metrics access and security headers.
#
#   scripts/container_smoke.sh swarm:test
set -euo pipefail

IMAGE="${1:-swarm:test}"
NAME="swarm-smoke-$$"
PORT="${PORT:-18080}"
TOKEN="smoke-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"
BASE="http://127.0.0.1:${PORT}"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; docker volume rm "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; docker logs "$NAME" 2>&1 | tail -40 >&2; exit 1; }

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" --read-only --tmpfs /tmp \
  --tmpfs /home/app:uid=10001,gid=10001 --cap-drop ALL --security-opt no-new-privileges:true \
  -e SWARM_API_TOKEN="$TOKEN" -e SWARM_ALLOW_MOCK=true -v "$NAME:/data" "$IMAGE" >/dev/null

for _ in $(seq 1 60); do curl -sf -o /dev/null "$BASE/healthz" && break; sleep 0.5; done
curl -sf "$BASE/healthz" >/dev/null || fail "healthz"
curl -sf "$BASE/readyz" | grep -q '"database":true' || fail "readyz"
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/jobs")" = 401 ] || fail "API open without a key"
[ "$(docker exec "$NAME" id -u)" = 10001 ] || fail "not running as uid 10001"

AUTH="Authorization: Bearer $TOKEN"
JOB=$(curl -sf -X POST "$BASE/api/research" -H "$AUTH" -H 'content-type: application/json' \
  -d '{"question":"Do heat pumps work in cold climates?","preset":"quick","mock":true}' \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["id"])') || fail "submit job"
STATUS=queued
for _ in $(seq 1 120); do
  STATUS=$(curl -sf "$BASE/api/jobs/$JOB" -H "$AUTH" | python3 -c 'import sys, json; print(json.load(sys.stdin)["status"])')
  case "$STATUS" in done|error|cancelled) break ;; esac
  sleep 0.5
done
[ "$STATUS" = done ] || fail "job ended as $STATUS"
curl -sf "$BASE/api/jobs/$JOB/report" -H "$AUTH" | grep -q "## Sources" || fail "report"
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/metrics")" = 401 ] || fail "metrics open without a key"
curl -sf "$BASE/metrics" -H "$AUTH" | grep -q 'swarm_jobs{kind="research",status="done"} 1' || fail "metrics"
curl -sI "$BASE/healthz" | grep -qi '^content-security-policy:' || fail "security headers"
docker exec "$NAME" swarm keys create --name smoke --daily-usd 1 >/dev/null 2>&1 || fail "keys create"
echo "container smoke test passed ($IMAGE, job $JOB)"
