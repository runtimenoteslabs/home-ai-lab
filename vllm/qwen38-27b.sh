#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 + DFlash2 draft on vLLM.
# Adapted from NVIDIA's sync-vllm-spark-128gb-qwen38-27b.sh
# (dgx-spark-playbooks). Same image and serve flags, except:
#   - models come from ~/models (read-only, offline), not the HF cache
#   - listens on the Tailscale IP with an API key, not 127.0.0.1
#   - runs as a detached container, not tied to an NVIDIA Sync session
# Normally started by systemd (ailab-vllm.service, SUPERVISED=1): the script then
# waits on the container and exits with its exit code, so systemd can restart on failure.
# Run by hand (no SUPERVISED), it just starts the container and returns.
# Labelled ailab.gpu=true so ailab-memwatch stops it before memory runs out.
set -euo pipefail

NAME="vllm-qwen38-27b"
HOST_IP="TAILSCALE_IP"   # ailab's Tailscale address
PORT=8000
MODELS="$HOME/models"
ENV_FILE="$HOME/ailab/vllm/env"   # contains VLLM_API_KEY=..., chmod 600
# Two-model setup (tested): 0.34 and 131072 so Qwen3.6 (ailab-vllm-qwen36, port 8001) fits beside it.
# Alone on the box: MEM=0.60 MAXLEN=262144 (and change the precheck in ailab-vllm.service back to 90).
MEM=0.34
MAXLEN=131072

[[ -r "$ENV_FILE" ]] || { echo "missing $ENV_FILE" >&2; exit 1; }
for d in Qwen3.8-27B-NVFP4 Qwen3.8-27B-DFlash2; do
  [[ -f "$MODELS/$d/config.json" ]] || { echo "missing $MODELS/$d (download not finished?)" >&2; exit 1; }
done

# keep the previous run's log before removing its container (crash post-mortems)
if docker inspect "$NAME" >/dev/null 2>&1; then
  mkdir -p "$HOME/ailab/logs/vllm"
  docker logs --tail 3000 "$NAME" > "$HOME/ailab/logs/vllm/$NAME-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  ls -1t "$HOME/ailab/logs/vllm/$NAME-"*.log 2>/dev/null | tail -n +21 | xargs -r rm -f   # keep the last 20
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --label ailab.gpu=true \
  --restart no \
  --oom-score-adj 500 \
  --log-opt max-size=50m --log-opt max-file=3 \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  --env-file "$ENV_FILE" \
  -e HF_HUB_OFFLINE=1 \
  -v "$MODELS:/models:ro" \
  nvcr.io/nvidia/vllm:26.10.pre1-py3 \
  vllm serve /models/Qwen3.8-27B-NVFP4 \
    --served-model-name qwen3.8-27b \
    --host "$HOST_IP" \
    --port "$PORT" \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAXLEN" \
    --max-num-seqs 8 \
    --async-scheduling \
    --enable-flashinfer-autotune \
    --gpu-memory-utilization "$MEM" \
    --kernel-config '{"linear_backend":"nvfp4_dynamic","nvfp4_dynamic_max_tokens":16,"unquantized_linear_backend":"flashinfer_cudnn","linear_backend_per_quant":{"fp8_w8a8":"cutlass"}}' \
    --speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' \
    --compilation-config '{"pass_config":{"fuse_act_quant":true},"max_cudagraph_capture_size":64}' \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice

echo "started $NAME on http://$HOST_IP:$PORT/v1  (logs: docker logs -f $NAME)"

if [[ "${SUPERVISED:-0}" == "1" ]]; then
  rc=$(docker wait "$NAME")
  echo "$NAME exited with code $rc"
  exit "$rc"
fi
echo "not supervised: stop with 'docker stop $NAME'. Normally use: sudo systemctl start|stop ailab-vllm"
