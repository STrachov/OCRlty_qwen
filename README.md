# OCRlty Qwen3‑VL Orchestrator

FastAPI service that calls a **vLLM OpenAI‑compatible endpoint** (or a local **mock backend**) to extract structured fields from receipt images using a JSON Schema–constrained response.

This repo also includes a small **SQLite‑based API key system** (roles + scopes) and a **debug stage** gated by both scopes and a global `DEBUG_MODE` flag.

---

## Features

- **/v1/extract**: send an image (URL or base64) → get a JSON object validated by `receipt_fields_v1.schema.json`
- **Artifacts**: every request is persisted as JSON under `ARTIFACTS_DIR/YYYY-MM-DD/<request_id>.json`
- **Auth with scopes** (SQLite):
  - `client`: `extract:run`
  - `debugger`: `extract:run`, `debug:run`
  - `admin`: `extract:run`, `debug:run`, `debug:read_raw`
- **Debug stage** (requires `DEBUG_MODE=1` + `debug:*` scopes)
  - prompt/temperature/max_tokens overrides
  - `POST /v1/debug/dry-run`: inspect the payload that would be sent to vLLM
- **Local testing without a model**
  - `INFERENCE_BACKEND=mock` returns deterministic fake model outputs
  - special request_id prefixes: `fail_json...`, `fail_schema...`

---

## Project layout (typical)

```
app/
  main.py            # FastAPI routes
  settings.py        # Settings via env / .env
  auth.py            # API keys + scopes (SQLite)
  auth_cli.py        # CLI for managing keys
  vllm_client.py     # minimal async client for vLLM OpenAI API
schemas/
  receipt_fields_v1.schema.json
prompts/
  receipt_fields_v1.txt
```

---

## Quickstart (local)

### 1) Create a virtualenv & install deps

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Create `.env` (example)

```bash
# --- Auth ---
AUTH_ENABLED=1
AUTH_DB_PATH=/workspace/auth/auth.db
API_KEY_PEPPER=change-me-to-a-long-random-secret

# --- Debug stage (global kill switch) ---
DEBUG_MODE=0

# --- Local modelless testing ---
INFERENCE_BACKEND=vllm   # or: mock

# --- Artifacts / model ---
ARTIFACTS_DIR=/workspace/artifacts
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# --- vLLM endpoint ---
VLLM_BASE_URL=http://127.0.0.1:8000
VLLM_API_KEY=
VLLM_TIMEOUT_S=120
```

> Notes:
> - If `AUTH_ENABLED=1`, `API_KEY_PEPPER` **must** be set.
> - Debug features are **disabled** unless `DEBUG_MODE=1` (even if your key has debug scopes).

### 3) Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Check health:

```bash
curl -s http://127.0.0.1:8080/health | jq
```

---

## Auth: manage API keys (SQLite)

The CLI is in `app/auth_cli.py`.

### Initialize the DB

```bash
python -m app.auth_cli init-db
```

### Create keys

```bash
# Client key (extract only)
python -m app.auth_cli create-key --key-id client-1 --role client

# Debugger key (extract + debug:run)
python -m app.auth_cli create-key --key-id dbg-1 --role debugger

# Admin key (extract + debug:run + debug:read_raw)
python -m app.auth_cli create-key --key-id admin-1 --role admin
```

You’ll see the secret **only once**:

```
=== API KEY CREATED (shown only once) ===
key_id:  client-1
role:    client
scopes:  ['extract:run']
api_key: <SECRET>
=======================================
```

### List keys

```bash
python -m app.auth_cli list-keys
```

### Revoke a key

```bash
python -m app.auth_cli revoke-key --key-id client-1
```

### Use the key in requests

Preferred header:

```bash
-H "Authorization: Bearer <SECRET>"
```

Fallback header:

```bash
-H "X-API-Key: <SECRET>"
```

---

## API usage

### 1) Who am I?

```bash
curl -s http://127.0.0.1:8080/v1/me \
  -H "Authorization: Bearer <SECRET>" | jq
```

### 2) Extract fields (image URL)

```bash
curl -s http://127.0.0.1:8080/v1/extract \
  -H "Authorization: Bearer <SECRET>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-001",
    "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
  }' | jq
```

### 3) Extract fields (base64 image)

```bash
# Linux/macOS example: create base64 without newlines
B64=$(base64 -w 0 receipt.png)

curl -s http://127.0.0.1:8080/v1/extract \
  -H "Authorization: Bearer <SECRET>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-002",
    "mime_type": "image/png",
    "image_base64": "'"$B64"'"
  }' | jq
```

---

## Debug stage (requires `DEBUG_MODE=1`)

### Enable debug

Set env:

```bash
export DEBUG_MODE=1
# or put DEBUG_MODE=1 into .env and restart the service
```

### A) Dry-run: inspect payload (no model call)

Requires scope `debug:run`.

```bash
curl -s http://127.0.0.1:8080/v1/debug/dry-run \
  -H "Authorization: Bearer <DEBUGGER_OR_ADMIN_SECRET>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "dryrun-001",
    "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
  }' | jq
```

### B) Override prompt / temperature / max_tokens

Send a `debug` block to `/v1/extract` (requires `debug:run`).

```bash
curl -s http://127.0.0.1:8080/v1/extract \
  -H "Authorization: Bearer <DEBUGGER_OR_ADMIN_SECRET>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "dbg-001",
    "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
    "debug": {
      "temperature": 0.0,
      "max_tokens": 128,
      "prompt_override": "Return JSON according to the schema. ..."
    }
  }' | jq
```

### Raw model text in responses

- The response field `raw_model_text` is returned **only if**:
  - `DEBUG_MODE=1`, **and**
  - the caller has scope `debug:read_raw` (role `admin` by default)

---

## Local testing without a model (mock backend)

### Enable mock backend

```bash
export INFERENCE_BACKEND=mock
# or INFERENCE_BACKEND=mock in .env and restart
```

Now `/v1/extract` won’t call vLLM.

### Deterministic failure modes

- `request_id` starting with `fail_json` → mock returns invalid JSON (`"{"`)
- `request_id` starting with `fail_schema` → mock returns `{}` (valid JSON but typically schema-invalid)

Example:

```bash
curl -s http://127.0.0.1:8080/v1/extract \
  -H "Authorization: Bearer <SECRET>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "fail_json-001",
    "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
  }' | jq
```

### Custom mock result (DEBUG only)

If `DEBUG_MODE=1` and you have `debug:run`, you can inject a custom JSON object that will be treated as the “model output”:

```bash
curl -s http://127.0.0.1:8080/v1/extract \
  -H "Authorization: Bearer <DEBUGGER_OR_ADMIN_SECRET>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "mock-override-001",
    "image_url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
    "debug": {
      "mock_result": {"total": 123.45, "currency": "EUR"}
    }
  }' | jq
```

---

## Artifacts

For each request (success or failure) the service writes an artifact JSON:

```
ARTIFACTS_DIR/YYYY-MM-DD/<request_id>.json
```

Artifacts typically include:
- request payload
- model id / backend
- raw response + parsed JSON
- schema validation errors
- timings

> The API response **does not** leak artifact paths unless raw output is allowed (`DEBUG_MODE=1` + `debug:read_raw`).

---

## Working with `requirements.in` (pip-tools)

Recommended workflow:
1) Edit **`requirements.in`** (top-level deps only)
2) Compile a pinned lockfile (**`requirements.txt`**)
3) Install exactly pinned deps

### Install pip-tools

```bash
pip install -U pip-tools
```

### Compile

```bash
pip-compile requirements.in -o requirements.txt
```

Optional:
- add `--upgrade` to refresh all pinned versions
- add `--resolver=backtracking` if needed for tricky resolution

### Sync environment to match the lockfile

```bash
pip-sync requirements.txt
```

> Commit both `requirements.in` and the generated `requirements.txt` so deployments are reproducible.

---

## PowerShell tips (Windows)

PowerShell’s `curl` is often an alias for `Invoke-WebRequest`, so headers behave differently.
Use `Invoke-RestMethod` like this:

```powershell
$headers = @{ Authorization = "Bearer <SECRET>" }
Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/me" -Headers $headers
```

---

## License / Disclaimer

This is an internal orchestrator template. Review and harden before exposing publicly (rate limits, quotas, network policies, logging redaction, etc.).
