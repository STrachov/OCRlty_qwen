# ocrlty-qwen3vl

Minimal production-oriented skeleton: runs vLLM (OpenAI-compatible) + a FastAPI orchestrator in a single container.

## Services
- vLLM: http://localhost:8000
- Orchestrator: http://localhost:8080

## Environment variables
- HF_TOKEN (required to download the model)
- HF_HOME (recommended, for caching)
- VLLM_MODEL (default: Qwen/Qwen3-VL-8B-Instruct)
- VLLM_PORT (default: 8000)
- VLLM_MAX_MODEL_LEN (default: 4096)
- VLLM_GPU_MEMORY_UTILIZATION (default: 0.90)
- VLLM_MAX_NUM_SEQS (default: 1)
- VLLM_ENFORCE_EAGER (default: 1)
- ARTIFACTS_DIR (default: /workspace/artifacts)

## Orchestrator endpoints
- GET /health
- POST /extract/receipt_fields_v1

Example request:
```bash
curl -s http://127.0.0.1:8080/extract/receipt_fields_v1 \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
  }'
Artifacts are saved under ARTIFACTS_DIR/YYYY-MM-DD/<request_id>.json