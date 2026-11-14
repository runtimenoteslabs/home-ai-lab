#!/usr/bin/env bash
# Checks run by the ailab model services before starting a model.
#   vllm-precheck.sh <need_gib> [<other-unit>:<other-port>]
#   ailab-vllm (Qwen3.8-27B, 0.34):       vllm-precheck.sh 65 ailab-vllm-qwen36:8001
#   ailab-vllm-qwen36 (Qwen3.6, 0.31):    vllm-precheck.sh 62 ailab-vllm:8000
# Two vLLM processes must not size their memory at the same time (each would count the other's
# allocations), so if the other model's service is running but not serving yet, wait for it.
set -u
TS_IP="TAILSCALE_IP"
NEED_GIB="${1:-90}"
OTHER="${2:-}"

if [ -e /run/ailab-memwatch.emergency ]; then
  echo "memwatch emergency marker present ($(cat /run/ailab-memwatch.emergency)); not starting. Reboot, or: sudo rm /run/ailab-memwatch.emergency"
  exit 1
fi

for i in $(seq 1 90); do
  ip -4 addr show tailscale0 2>/dev/null | grep -q "inet ${TS_IP}/" && break
  [ "$i" -eq 90 ] && { echo "Tailscale address ${TS_IP} not up after 3 min"; exit 1; }
  sleep 2
done

if [ -n "$OTHER" ]; then
  unit="${OTHER%%:*}"; port="${OTHER##*:}"
  key=$(grep '^VLLM_API_KEY=' "$HOME/ailab/vllm/env" | cut -d= -f2-)
  for i in $(seq 1 180); do
    systemctl is-active --quiet "$unit" || break                       # other model not running: go
    curl -sf -m 5 -H "Authorization: Bearer $key" "http://${TS_IP}:${port}/v1/models" >/dev/null && break   # serving: go
    [ "$i" -eq 1 ] && echo "waiting for $unit to finish loading..."
    [ "$i" -eq 180 ] && { echo "$unit still loading after 15 min"; exit 1; }
    sleep 5
  done
fi

a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$a" -lt "$NEED_GIB" ]; then
  echo "only ${a} GiB MemAvailable, need ${NEED_GIB}. Is another (bigger) model running?"
  exit 1
fi
echo "precheck ok: Tailscale up, ${a} GiB available (need ${NEED_GIB})"
