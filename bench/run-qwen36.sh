#!/usr/bin/env bash
# Test container for Qwen3.6-35B-A3B NVFP4. Mia's settings in NVIDIA's vLLM image
# Not a service: --restart no, memwatch can stop it (label ailab.gpu=true).
#   alone:        MEM=0.60 ~/ailab/bench/run-qwen36.sh
#   beside 27B:   MEM=0.31 MAXLEN=131072 ~/ailab/bench/run-qwen36.sh   (Qwen3.8-27B lowered first)
#   stop:         docker stop -t 30 bench-qwen36
set -euo pipefail
NAME="bench-qwen36"
HOST_IP="TAILSCALE_IP"
PORT="${PORT:-8001}"
MEM="${MEM:-0.60}"
MOE="${MOE:-auto}"            # keep auto: flashinfer_b12x fails at start in vLLM 0.31 (FP8 MoE layers)
MAXLEN="${MAXLEN:-262144}"    # shared with Qwen3.8-27B: 131072
MODELS="$HOME/models"
ENV_FILE="$HOME/ailab/vllm/env"
LOGS="$HOME/ailab/logs/bench"
IMAGE="nvcr.io/nvidia/vllm:26.10.pre1-py3"

[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE"; exit 1; }
[[ -f "$MODELS/Qwen3.6-35B-A3B-NVFP4/config.json" ]] || { echo "model missing"; exit 1; }
mkdir -p "$LOGS"
if docker inspect "$NAME" >/dev/null 2>&1; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true   # graceful stop first (killing leaks shared memory with --ipc host)
  docker logs "$NAME" > "$LOGS/$NAME-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null
fi

docker run -d --name "$NAME" --label ailab.gpu=true --restart no --oom-score-adj 500 \
  --log-opt max-size=50m --log-opt max-file=3 \
  --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint "" --env-file "$ENV_FILE" \
  -e HF_HUB_OFFLINE=1 -e CUTE_DSL_ARCH=sm_121a \
  -e MAX_JOBS=4 -e NVCC_THREADS=2 -e FLASHINFER_NVCC_THREADS=2 \
  -v "$MODELS:/models:ro" \
  "$IMAGE" \
  vllm serve /models/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name qwen3.6-35b-a3b \
    --host "$HOST_IP" --port "$PORT" \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN" --max-num-seqs 8 --max-num-batched-tokens 32768 \
    --kv-cache-dtype fp8 --async-scheduling --enable-chunked-prefill \
    --moe-backend "$MOE" --linear-backend flashinfer_b12x --attention-backend flashinfer \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
echo "started $NAME on $HOST_IP:$PORT (gpu-memory-utilization $MEM, moe $MOE, max-model-len $MAXLEN)"
