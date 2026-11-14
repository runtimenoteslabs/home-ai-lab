#!/usr/bin/env bash
# Test container for Qwen3.8-27B NVFP4 + DFlash2: the production flags from
# ~/ailab/vllm/qwen38-27b.sh, with memory share and context adjustable. Not a service
# (--restart no); the production service ailab-vllm must be stopped first (same port).
#   MEM=0.34 MAXLEN=131072 ~/ailab/bench/run-qwen38.sh
#   stop: docker stop -t 30 bench-qwen38
set -euo pipefail
NAME="bench-qwen38"
HOST_IP="TAILSCALE_IP"
PORT="${PORT:-8000}"
MEM="${MEM:-0.34}"
MAXLEN="${MAXLEN:-131072}"
MODELS="$HOME/models"
ENV_FILE="$HOME/ailab/vllm/env"
LOGS="$HOME/ailab/logs/bench"

systemctl is-active --quiet ailab-vllm && { echo "ailab-vllm is running; stop it first"; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE"; exit 1; }
for d in Qwen3.8-27B-NVFP4 Qwen3.8-27B-DFlash2; do
  [[ -f "$MODELS/$d/config.json" ]] || { echo "missing $MODELS/$d"; exit 1; }
done
mkdir -p "$LOGS"
if docker inspect "$NAME" >/dev/null 2>&1; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker logs "$NAME" > "$LOGS/$NAME-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null
fi

docker run -d --name "$NAME" --label ailab.gpu=true --restart no --oom-score-adj 500 \
  --log-opt max-size=50m --log-opt max-file=3 \
  --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint "" --env-file "$ENV_FILE" -e HF_HUB_OFFLINE=1 \
  -e MAX_JOBS=4 -e NVCC_THREADS=2 -e FLASHINFER_NVCC_THREADS=2 \
  -v "$MODELS:/models:ro" \
  nvcr.io/nvidia/vllm:26.10.pre1-py3 \
  vllm serve /models/Qwen3.8-27B-NVFP4 \
    --served-model-name qwen3.8-27b \
    --host "$HOST_IP" --port "$PORT" \
    --kv-cache-dtype fp8 --max-model-len "$MAXLEN" --max-num-seqs 8 \
    --async-scheduling --enable-flashinfer-autotune \
    --gpu-memory-utilization "$MEM" \
    --kernel-config '{"linear_backend":"nvfp4_dynamic","nvfp4_dynamic_max_tokens":16,"unquantized_linear_backend":"flashinfer_cudnn","linear_backend_per_quant":{"fp8_w8a8":"cutlass"}}' \
    --speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' \
    --compilation-config '{"pass_config":{"fuse_act_quant":true},"max_cudagraph_capture_size":64}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
echo "started $NAME on $HOST_IP:$PORT (gpu-memory-utilization $MEM, max-model-len $MAXLEN)"
