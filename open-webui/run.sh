#!/usr/bin/env bash
# Open WebUI on ailab. Listens only on the Tailscale IP (port 3000), talks to vLLM on :8000.
# Settings and secrets: ~/ailab/open-webui/env (chmod 600). Data: ~/ailab/open-webui/data.
set -euo pipefail
NAME="open-webui"
IMAGE="ghcr.io/open-webui/open-webui:v0.13.3"
BASE="$HOME/ailab/open-webui"

[[ -r "$BASE/env" ]] || { echo "missing $BASE/env" >&2; exit 1; }
mkdir -p "$BASE/data"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --log-opt max-size=20m --log-opt max-file=3 \
  --network host \
  --env-file "$BASE/env" \
  --health-cmd 'curl -sf http://TAILSCALE_IP:3000/health || exit 1' \
  -v "$BASE/data:/app/backend/data" \
  "$IMAGE"
echo "started $NAME on http://TAILSCALE_IP:3000  (logs: docker logs -f $NAME)"
