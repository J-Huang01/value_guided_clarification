#!/bin/bash
set -euo pipefail
DATASET=${DATASET:?set DATASET}
AGENT_MODEL=${AGENT_MODEL:?set AGENT_MODEL}
USER_MODEL=${USER_MODEL:?set USER_MODEL}
AGENT_DEVICES=${AGENT_DEVICES:-0}
USER_DEVICES=${USER_DEVICES:-1}
AGENT_PORT=${AGENT_PORT:-8000}
USER_PORT=${USER_PORT:-8100}
OUT=${OUT:-runs/$DATASET}
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$AGENT_DEVICES VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve "$AGENT_MODEL" --port "$AGENT_PORT" \
    --gpu-memory-utilization 0.4 --max-num-batched-tokens 4096 --enable-lora --max-loras 2 ${AGENT_SERVE_ARGS:-} \
    > "$OUT/agent_server.log" 2>&1 &
AGENT_PID=$!
CUDA_VISIBLE_DEVICES=$USER_DEVICES vllm serve "$USER_MODEL" --port "$USER_PORT" --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes ${USER_SERVE_ARGS:-} \
    > "$OUT/user_server.log" 2>&1 &
USER_PID=$!
trap 'kill $AGENT_PID $USER_PID 2>/dev/null || true' EXIT
for port in "$AGENT_PORT" "$USER_PORT"; do
    until curl -sf "http://localhost:$port/v1/models" > /dev/null; do
        kill -0 $AGENT_PID 2>/dev/null && kill -0 $USER_PID 2>/dev/null || { echo "a model server exited, see $OUT/*_server.log"; exit 1; }
        sleep 10
    done
done
COMMON=(--dataset "$DATASET" --model "$AGENT_MODEL" --user-model "$USER_MODEL"
        --agent-url "http://localhost:$AGENT_PORT/v1" --user-url "http://localhost:$USER_PORT/v1")
CUDA_VISIBLE_DEVICES=$AGENT_DEVICES python cleat/train.py "${COMMON[@]}" --out "$OUT" "$@"
CUDA_VISIBLE_DEVICES=$AGENT_DEVICES python cleat/evaluate.py "${COMMON[@]}" --checkpoint "$OUT/checkpoints/final-refit" \
    --out "$OUT/test" ${EVAL_ARGS:-}
