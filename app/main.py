import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from jsonschema import Draft202012Validator

from app.vllm_client import VLLMClient
from app.auth import ApiPrincipal, init_auth_db, require_api_key
from app.settings import settings

APP_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = APP_ROOT / "schemas"
PROMPTS_DIR = APP_ROOT / "prompts"

ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR).resolve()
MODEL_ID = settings.VLLM_MODEL

# Load schema once at startup
RECEIPT_FIELDS_V1_SCHEMA_PATH = SCHEMAS_DIR / "receipt_fields_v1.schema.json"
RECEIPT_FIELDS_V1_SCHEMA = json.loads(RECEIPT_FIELDS_V1_SCHEMA_PATH.read_text(encoding="utf-8"))
RECEIPT_FIELDS_V1_VALIDATOR = Draft202012Validator(RECEIPT_FIELDS_V1_SCHEMA)

PROMPT_RECEIPT_FIELDS_V1 = (PROMPTS_DIR / "receipt_fields_v1.txt").read_text(encoding="utf-8").strip()

vllm = VLLMClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_auth_db()
    yield

app = FastAPI(
    title="Qwen3-VL Orchestrator", 
    version="0.1.0",
    lifespan=lifespan,
    )

class ExtractReceiptFieldsRequest(BaseModel):
    # Provide either image_url OR image_base64.
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mime_type: str = Field(default="image/png", description="Used only with image_base64.")
    request_id: Optional[str] = None


class ExtractReceiptFieldsResponse(BaseModel):
    request_id: str
    model: str
    schema_ok: bool
    result: Dict[str, Any]
    raw_model_text: str
    timings_ms: Dict[str, float]
    schema_errors: Optional[list[str]] = None


def _make_request_id(user_request_id: Optional[str]) -> str:
    return user_request_id or uuid.uuid4().hex


def _ensure_artifacts_dir() -> Path:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_artifact(request_id: str, payload: Dict[str, Any]) -> str:
    out_dir = _ensure_artifacts_dir()
    path = out_dir / f"{request_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _image_content(image_url: Optional[str], image_base64: Optional[str], mime_type: str) -> Dict[str, Any]:
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    if image_base64:
        # OpenAI-style data URL.
        data_url = f"data:{mime_type};base64,{image_base64}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    raise HTTPException(status_code=400, detail="Provide either image_url or image_base64.")


@app.get("/health")
async def health() -> Dict[str, Any]:
    # Simple orchestrator health + vLLM connectivity check.
    try:
        models = await vllm.models()
        return {"ok": True, "vllm_ok": True, "models": [m["id"] for m in models.get("data", [])]}
    except Exception as e:
        return {"ok": True, "vllm_ok": False, "error": str(e)}

@app.get("/v1/me")
async def me(principal: ApiPrincipal = Depends(require_api_key)):
    return {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))}

@app.post("/v1/extract", response_model=ExtractReceiptFieldsResponse)
async def extract_receipt_fields(
    req: ExtractReceiptFieldsRequest,
    principal: ApiPrincipal = Depends(require_api_key),  # <-- Stage 1: key required (if AUTH_ENABLED=1)
) -> ExtractReceiptFieldsResponse:
    request_id = _make_request_id(req.request_id)

    t0 = time.perf_counter()

    messages = [
        {
            "role": "user",
            "content": [
                _image_content(req.image_url, req.image_base64, req.mime_type),
                {"type": "text", "text": PROMPT_RECEIPT_FIELDS_V1},
            ],
        }
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "receipt_fields_v1",
            "schema": RECEIPT_FIELDS_V1_SCHEMA,
        },
    }

    try:
        t_vllm0 = time.perf_counter()
        raw = await vllm.chat_completions(
            model=MODEL_ID,
            messages=messages,
            temperature=0.0,
            max_tokens=128,
            response_format=response_format,
        )
        t_vllm1 = time.perf_counter()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vLLM request failed: {e}")

    try:
        raw_text = raw["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail="Unexpected vLLM response format.")

    # Parse JSON produced by the model.
    try:
        parsed = json.loads(raw_text)
    except Exception as e:
        artifact_path = _save_artifact(
            request_id,
            {
                "request_id": request_id,
                "model": MODEL_ID,
                "input": req.model_dump(),
                "auth": {"key_id": principal.key_id, "role": principal.role},  # safe metadata
                "raw_response": raw,
                "raw_model_text": raw_text,
                "error": f"json_parse_error: {e}",
                "timings_ms": {
                    "vllm_ms": (t_vllm1 - t_vllm0) * 1000.0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
            },
        )
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON. Artifact: {artifact_path}")

    # Validate against schema (double safety).
    errors = [e.message for e in RECEIPT_FIELDS_V1_VALIDATOR.iter_errors(parsed)]
    schema_ok = len(errors) == 0

    artifact_path = _save_artifact(
        request_id,
        {
            "request_id": request_id,
            "model": MODEL_ID,
            "input": req.model_dump(),
            "auth": {"key_id": principal.key_id, "role": principal.role},  # safe metadata
            "schema": "receipt_fields_v1",
            "schema_ok": schema_ok,
            "schema_errors": errors,
            "raw_model_text": raw_text,
            "parsed": parsed,
            "raw_response": raw,
            "timings_ms": {
                "vllm_ms": (t_vllm1 - t_vllm0) * 1000.0,
                "total_ms": (time.perf_counter() - t0) * 1000.0,
            },
        },
    )

    return ExtractReceiptFieldsResponse(
        request_id=request_id,
        model=MODEL_ID,
        schema_ok=schema_ok,
        result=parsed,
        raw_model_text=raw_text,
        timings_ms={
            "vllm_ms": (t_vllm1 - t_vllm0) * 1000.0,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        },
        schema_errors=errors if not schema_ok else None,
    )
