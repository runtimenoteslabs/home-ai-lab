#!/usr/bin/env bash
# Start a test model, wait until it's ready (memory guard: stop it below 15 GiB free), then benchmark it.
#   g-run.sh <label> <container> <served-model-name> <port> "<notes>" <launch command...>
# e.g. g-run.sh qwen36-moe-b12x bench-qwen36 qwen3.6-35b-a3b 8001 "moe flashinfer_b12x" \
#        env MEM=0.60 MOE=flashinfer_b12x ~/ailab/bench/run-qwen36.sh
set -uo pipefail
LABEL=$1 NAME=$2 SERVED=$3 PORT=$4 NOTES=$5; shift 5
B="$HOME/ailab/bench"
URL="http://TAILSCALE_IP:$PORT"
KEY=$(grep '^VLLM_API_KEY=' "$HOME/ailab/vllm/env" | cut -d= -f2-)
avail_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
log() { echo "$(date +%H:%M:%S) $*"; }
end() { echo "RUN-END $LABEL"; exit "${1:-0}"; }

log "$LABEL: MemAvailable $(avail_gib) GiB before launch"
t0=$(date +%s)
"$@" || { log "launch failed"; end 1; }
min=999
while :; do
  a=$(avail_gib); [ "$a" -lt "$min" ] && min=$a
  if [ "$a" -lt 15 ]; then log "ABORT: MemAvailable $a GiB < 15"; docker stop -t 30 "$NAME"; end 1; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then log "container exited during startup:"; docker logs --tail 40 "$NAME" 2>&1; end 1; fi
  curl -sf -H "Authorization: Bearer $KEY" "$URL/v1/models" >/dev/null && break
  if [ $(( $(date +%s) - t0 )) -gt 2400 ]; then log "not ready after 40 min"; docker logs --tail 40 "$NAME" 2>&1; end 1; fi
  sleep 3
done
ready=$(( $(date +%s) - t0 ))
log "ready after ${ready}s; lowest MemAvailable during startup ${min} GiB; now $(avail_gib) GiB"
docker logs "$NAME" 2>&1 | grep -E "Model loading took|Available KV cache memory|GPU KV cache size|NvFp4 MoE backend|no kernel for this linear" | sed 's/^(EngineCore[^)]*) //' | cut -c1-200 | sort -u
python3 "$B/ailab-bench.py" --label "$LABEL" --base-url "$URL/v1" --model "$SERVED" \
  --notes "$NOTES; startup ${ready}s, startup min MemAvailable ${min} GiB"
log "bench exit $?"
end 0
