#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (env-overridable)
# ----------------------------
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_LIMIT_MM_VIDEO="${VLLM_LIMIT_MM_VIDEO:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"   # 1/0

ORCH_HOST="${ORCH_HOST:-0.0.0.0}"
ORCH_PORT="${ORCH_PORT:-8080}"

# Optional API key (if set, vLLM will require Authorization: Bearer ...)
VLLM_API_KEY="${VLLM_API_KEY:-}"

# ----------------------------
# Start vLLM (background)
# ----------------------------
echo "[entrypoint] Starting vLLM..."
args=(
  "serve" "${VLLM_MODEL}"
  "--host" "${VLLM_HOST}"
  "--port" "${VLLM_PORT}"
  "--dtype" "${VLLM_DTYPE}"
  "--max-model-len" "${VLLM_MAX_MODEL_LEN}"
  "--gpu-memory-utilization" "${VLLM_GPU_MEMORY_UTILIZATION}"
  "--max-num-seqs" "${VLLM_MAX_NUM_SEQS}"
  "--limit-mm-per-prompt.video" "${VLLM_LIMIT_MM_VIDEO}"
)

if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  args+=("--enforce-eager")
fi

if [[ -n "${VLLM_API_KEY}" ]]; then
  args+=("--api-key" "${VLLM_API_KEY}")
fi

# Run vLLM in background
vllm "${args[@]}" &
VLLM_PID=$!

# Ensure we kill vLLM on exit
cleanup() {
  echo "[entrypoint] Shutting down..."
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -TERM "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

# ----------------------------
# Wait for vLLM readiness
# ----------------------------
echo "[entrypoint] Waiting for vLLM to become ready..."
deadline=$((SECONDS + 300))
while true; do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[entrypoint] vLLM is ready."
    break
  fi
  if (( SECONDS > deadline )); then
    echo "[entrypoint] ERROR: vLLM did not become ready in time."
    exit 1
  fi
  sleep 1
done

# ----------------------------
# Start orchestrator (foreground)
# ----------------------------
echo "[entrypoint] Starting orchestrator on ${ORCH_HOST}:${ORCH_PORT}..."
exec uvicorn app.main:app --host "${ORCH_HOST}" --port "${ORCH_PORT}"
