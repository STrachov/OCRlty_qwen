#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] ===== Qwen3-VL container start ====="

# ----------------------------
# HF cache on /workspace (network volume)
# ----------------------------
export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
mkdir -p "${HF_HOME}"
echo "[entrypoint] HF_HOME=${HF_HOME}"

# ----------------------------
# Code sync (git clone/pull) – same idea as your previous project
# ----------------------------
APP_REPO_URL="${APP_REPO_URL:-https://github.com/STrachov/OCRlty_qwen.git}"
APP_SRC_ROOT="${APP_SRC_ROOT:-/workspace/src}"   # git working copy on volume
APP_FALLBACK_ROOT="${APP_FALLBACK_ROOT:-/opt/app}"  # baked code
APP_ROOT="${APP_FALLBACK_ROOT}"

echo "[entrypoint] ===== Code selection phase ====="
echo "[entrypoint] APP_REPO_URL=${APP_REPO_URL}"
echo "[entrypoint] APP_SRC_ROOT=${APP_SRC_ROOT}"
echo "[entrypoint] Fallback APP_ROOT=${APP_FALLBACK_ROOT}"

if command -v git >/dev/null 2>&1; then
  if [ -d "${APP_SRC_ROOT}/.git" ]; then
    echo "[entrypoint] Found existing git repo at ${APP_SRC_ROOT}, running git pull..."
    (
      cd "${APP_SRC_ROOT}" && \
      git pull --ff-only || echo "[entrypoint] git pull failed, keeping existing code"
    )
    APP_ROOT="${APP_SRC_ROOT}"
  else
    if [ ! -d "${APP_SRC_ROOT}" ] || [ -z "$(ls -A "${APP_SRC_ROOT}" 2>/dev/null || true)" ]; then
      echo "[entrypoint] Cloning repo into ${APP_SRC_ROOT}..."
      mkdir -p "${APP_SRC_ROOT}"
      if git clone --depth 1 "${APP_REPO_URL}" "${APP_SRC_ROOT}"; then
        echo "[entrypoint] Clone OK."
        APP_ROOT="${APP_SRC_ROOT}"
      else
        echo "[entrypoint] git clone FAILED, using baked code at ${APP_FALLBACK_ROOT}"
        APP_ROOT="${APP_FALLBACK_ROOT}"
      fi
    else
      echo "[entrypoint] ${APP_SRC_ROOT} not empty and not a git repo, using baked code."
      APP_ROOT="${APP_FALLBACK_ROOT}"
    fi
  fi
else
  echo "[entrypoint] git not found, using baked code at ${APP_FALLBACK_ROOT}"
  APP_ROOT="${APP_FALLBACK_ROOT}"
fi

echo "[entrypoint] Final APP_ROOT=${APP_ROOT}"

# Optional debug sleep
if [ "${SLEEP_ON_START:-0}" = "1" ]; then
  echo "[entrypoint] SLEEP_ON_START=1 → sleeping indefinitely for debug..."
  exec tail -f /dev/null
fi

# Ensure imports work regardless of where code lives
export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"

# ----------------------------
# vLLM config (env-overridable)
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
VLLM_API_KEY="${VLLM_API_KEY:-}"

# Orchestrator config
ORCH_HOST="${ORCH_HOST:-0.0.0.0}"
ORCH_PORT="${ORCH_PORT:-8080}"
ORCH_RELOAD="${ORCH_RELOAD:-1}"  # 1 enables hot reload (useful with git pull)
ORCH_APP="${ORCH_APP:-app.main:app}"

echo "[entrypoint] ===== Starting vLLM ====="
echo "[entrypoint] VLLM_MODEL=${VLLM_MODEL}"
echo "[entrypoint] VLLM_PORT=${VLLM_PORT}"
echo "[entrypoint] ORCH_RELOAD=${ORCH_RELOAD}"

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

vllm "${args[@]}" &
VLLM_PID=$!

cleanup() {
  echo "[entrypoint] Shutting down..."
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -TERM "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

# Wait for vLLM readiness
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

echo "[entrypoint] ===== Starting orchestrator ====="
cd "${APP_ROOT}"

if [[ "${ORCH_RELOAD}" == "1" ]]; then
  echo "[entrypoint] ORCH_RELOAD=1 (hot reload enabled)"
  exec uvicorn "${ORCH_APP}" --host "${ORCH_HOST}" --port "${ORCH_PORT}" --reload --reload-dir "${APP_ROOT}"
else
  exec uvicorn "${ORCH_APP}" --host "${ORCH_HOST}" --port "${ORCH_PORT}"
fi
