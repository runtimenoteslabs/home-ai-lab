#!/usr/bin/env bash
# Qwen3.6-35B-A3B NVFP4 (Unsloth, revision 739af1e7aac3) with its built-in MTP draft, on vLLM.
# Mia's settings (MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark) in NVIDIA's image.
# Normally started by systemd (ailab-vllm-qwen36.service, SUPERVISED=1), which waits for Qwen3.8-27B
# to finish loading first. Labelled ailab.gpu=true so ailab-memwatch stops it before memory runs out.
# Sampling: the model's own defaults (temperature 1.0, top_p 0.95, top_k 20 = Qwen's "thinking, general").
set -euo pipefail

NAME="vllm-qwen36"
HOST_IP="TAILSCALE_IP"   # ailab's Tailscale address
PORT=8001
MODELS="$HOME/models"
ENV_FILE="$HOME/ailab/vllm/env"   # same API key as port 8000
MEM=0.31          # beside Qwen3.8-27B at 0.34; alone on the box 0.60 is plenty
MAXLEN=131072

[[ -r "$ENV_FILE" ]] || { echo "missing $ENV_FILE" >&2; exit 1; }
[[ -f "$MODELS/Qwen3.6-35B-A3B-NVFP4/config.json" ]] || { echo "missing $MODELS/Qwen3.6-35B-A3B-NVFP4" >&2; exit 1; }

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
  -e CUTE_DSL_ARCH=sm_121a \
  -e MAX_JOBS=4 -e NVCC_THREADS=2 -e FLASHINFER_NVCC_THREADS=2 \
  -v "$MODELS:/models:ro" \
  nvcr.io/nvidia/vllm:26.10.pre1-py3 \
  vllm serve /models/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name qwen3.6-35b-a3b \
    --host "$HOST_IP" \
    --port "$PORT" \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs 8 \
    --max-num-batched-tokens 32768 \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --enable-chunked-prefill \
    --moe-backend auto \
    --linear-backend flashinfer_b12x \
    --attention-backend flashinfer \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice

echo "started $NAME on http://$HOST_IP:$PORT/v1  (logs: docker logs -f $NAME)"

if [[ "${SUPERVISED:-0}" == "1" ]]; then
  rc=$(docker wait "$NAME")
  echo "$NAME exited with code $rc"
  exit "$rc"
fi
echo "not supervised: stop with 'docker stop $NAME'. Normally use: sudo systemctl start|stop ailab-vllm-qwen36"
