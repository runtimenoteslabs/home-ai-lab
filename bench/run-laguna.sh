#!/usr/bin/env bash
# Test container for Laguna S 2.1 NVFP4 + DFlash draft, NVIDIA's vLLM image, which
# supports Laguna natively (no --trust-remote-code; the repo's .py files are never run).
# Flags from Poolside's card ("Maximum performance on the DGX Spark") and Mia's recipe.
#   MEM=0.70 ~/ailab/bench/run-laguna.sh        stop: docker stop -t 30 bench-laguna
set -euo pipefail
NAME="bench-laguna"
HOST_IP="TAILSCALE_IP"
PORT="${PORT:-8001}"
MEM="${MEM:-0.70}"
MAXLEN="${MAXLEN:-262144}"
MODELS="$HOME/models"
ENV_FILE="$HOME/ailab/vllm/env"
LOGS="$HOME/ailab/logs/bench"

[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE"; exit 1; }
for d in Laguna-S-2.1-NVFP4 Laguna-S-2.1-DFlash-NVFP4; do
  [[ -f "$MODELS/$d/config.json" ]] || { echo "missing $MODELS/$d"; exit 1; }
done
mkdir -p "$LOGS"
if docker inspect "$NAME" >/dev/null 2>&1; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker logs "$NAME" > "$LOGS/$NAME-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null
fi

# MAX_JOBS / NVCC_THREADS cap kernel compiles: Poolside's card warns an uncapped nvcc fan-out
# on a cold cache can use up the 128 GB and take the box down.
docker run -d --name "$NAME" --label ailab.gpu=true --restart no --oom-score-adj 500 \
  --log-opt max-size=50m --log-opt max-file=3 \
  --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint "" --env-file "$ENV_FILE" \
  -e HF_HUB_OFFLINE=1 -e CUTE_DSL_ARCH=sm_121a \
  -e MAX_JOBS=4 -e NVCC_THREADS=2 -e FLASHINFER_NVCC_THREADS=2 \
  -v "$MODELS:/models:ro" \
  nvcr.io/nvidia/vllm:26.10.pre1-py3 \
  vllm serve /models/Laguna-S-2.1-NVFP4 \
    --served-model-name laguna-s-2.1 \
    --host "$HOST_IP" --port "$PORT" \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN" --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --speculative-config '{"method":"dflash","model":"/models/Laguna-S-2.1-DFlash-NVFP4","num_speculative_tokens":7}' \
    --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --enable-auto-tool-choice
echo "started $NAME on $HOST_IP:$PORT (gpu-memory-utilization $MEM, max-model-len $MAXLEN)"
