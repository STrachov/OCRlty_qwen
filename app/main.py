import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from jsonschema import Draft202012Validator

from app.auth import ApiPrincipal, init_auth_db, require_scopes
from app.settings import settings
from app.vllm_client import VLLMClient

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


class DebugOptions(BaseModel):
    # Any debug usage requires DEBUG_MODE=1 (global env switch).
    prompt_override: Optional[str] = Field(default=None, description="Override the base prompt (DEBUG only).")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override temperature (DEBUG only).")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Override max_tokens (DEBUG only).")

    # Mock backend controls (DEBUG only).
    mock_mode: Optional[str] = Field(
        default=None,
        description="For INFERENCE_BACKEND=mock: 'ok' | 'fail_json' | 'fail_schema'. "
                    "If omitted, may be inferred from request_id prefix.",
    )
    mock_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For INFERENCE_BACKEND=mock: provide a JSON object to return as the model output (DEBUG only).",
    )


class ExtractReceiptFieldsRequest(BaseModel):
    # Provide either image_url OR image_base64.
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mime_type: str = Field(default="image/png", description="Used only with image_base64.")
    request_id: Optional[str] = None

    # Debug block. Presence of this block requires DEBUG_MODE=1 AND debug:run scope.
    debug: Optional[DebugOptions] = None


class ExtractReceiptFieldsResponse(BaseModel):
    request_id: str
    model: str
    schema_ok: bool
    result: Dict[str, Any]
    timings_ms: Dict[str, float]
    schema_errors: Optional[list[str]] = None

    # Returned only when DEBUG_MODE=1 and caller has debug:read_raw (or AUTH is disabled).
    raw_model_text: Optional[str] = None


class DryRunResponse(BaseModel):
    request_id: str
    model: str
    inference_backend: str
    temperature: float
    max_tokens: int
    messages: Any
    response_format: Any


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
        data_url = f"data:{mime_type};base64,{image_base64}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    raise HTTPException(status_code=400, detail="Provide either image_url or image_base64.")


def _debug_enabled_or_403() -> None:
    if not settings.DEBUG_MODE:
        raise HTTPException(status_code=403, detail="Debug features are disabled (DEBUG_MODE=0).")


def _debug_allowed(principal: ApiPrincipal) -> None:
    _debug_enabled_or_403()
    if settings.AUTH_ENABLED and ("debug:run" not in principal.scopes):
        raise HTTPException(status_code=403, detail="Missing required scope: debug:run")


def _raw_allowed(principal: ApiPrincipal) -> bool:
    if not settings.DEBUG_MODE:
        return False
    if not settings.AUTH_ENABLED:
        return True
    return "debug:read_raw" in principal.scopes


def _clamp_max_tokens(v: int) -> int:
    return max(1, min(int(v), int(settings.MAX_TOKENS_CAP)))


def _apply_debug_overrides(
    principal: ApiPrincipal,
    debug: Optional[DebugOptions],
) -> Tuple[str, float, int]:
    prompt_text = PROMPT_RECEIPT_FIELDS_V1
    temperature = 0.0
    max_tokens = 128

    if debug is None:
        return prompt_text, temperature, max_tokens

    _debug_allowed(principal)

    if debug.prompt_override is not None:
        p = debug.prompt_override.strip()
        if len(p) > int(settings.MAX_PROMPT_CHARS):
            raise HTTPException(status_code=400, detail=f"prompt_override too long (>{settings.MAX_PROMPT_CHARS} chars).")
        if p:
            prompt_text = p

    if debug.temperature is not None:
        temperature = float(debug.temperature)

    if debug.max_tokens is not None:
        max_tokens = int(debug.max_tokens)

    max_tokens = _clamp_max_tokens(max_tokens)
    return prompt_text, temperature, max_tokens


def _resolve_ref(ref: str, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    cur: Any = root_schema
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return {}
    return cur if isinstance(cur, dict) else {}


def _sample_from_schema(schema: Dict[str, Any], root_schema: Dict[str, Any]) -> Any:
    if "$ref" in schema and isinstance(schema["$ref"], str):
        resolved = _resolve_ref(schema["$ref"], root_schema)
        if resolved:
            return _sample_from_schema(resolved, root_schema)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            return _sample_from_schema(schema[key][0], root_schema)

    t = schema.get("type")
    if isinstance(t, list) and t:
        t = t[0]

    if t == "object" or ("properties" in schema):
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        out: Dict[str, Any] = {}
        for k in required:
            if k in props and isinstance(props[k], dict):
                out[k] = _sample_from_schema(props[k], root_schema)
            else:
                out[k] = None
        added = 0
        for k, v in props.items():
            if k in out:
                continue
            if isinstance(v, dict):
                out[k] = _sample_from_schema(v, root_schema)
                added += 1
            if added >= 2:
                break
        return out

    if t == "array":
        items = schema.get("items", {})
        min_items = int(schema.get("minItems", 0) or 0)
        if min_items > 0 and isinstance(items, dict):
            return [_sample_from_schema(items, root_schema)]
        return []

    if t == "string":
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            return str(schema["enum"][0])
        fmt = schema.get("format")
        if fmt == "date":
            return "2026-01-24"
        if fmt == "date-time":
            return "2026-01-24T12:00:00Z"
        return "mock"

    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False

    return None


def _mock_chat_completions(
    request_id: str,
    principal: ApiPrincipal,
    debug: Optional[DebugOptions],
) -> Dict[str, Any]:
    mode = "ok"
    if request_id.startswith("fail_json"):
        mode = "fail_json"
    elif request_id.startswith("fail_schema"):
        mode = "fail_schema"

    if debug is not None:
        _debug_allowed(principal)
        if debug.mock_mode:
            mode = debug.mock_mode.strip().lower()
        if debug.mock_result is not None:
            content = json.dumps(debug.mock_result, ensure_ascii=False)
            return {"choices": [{"message": {"content": content}}]}

    if mode == "fail_json":
        content = "{"
    elif mode == "fail_schema":
        content = "{}"
    else:
        sample = _sample_from_schema(RECEIPT_FIELDS_V1_SCHEMA, RECEIPT_FIELDS_V1_SCHEMA)
        content = json.dumps(sample, ensure_ascii=False)

    return {"choices": [{"message": {"content": content}}]}


def _build_messages(prompt_text: str, req: ExtractReceiptFieldsRequest) -> Any:
    return [
        {
            "role": "user",
            "content": [
                _image_content(req.image_url, req.image_base64, req.mime_type),
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _build_response_format() -> Any:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "receipt_fields_v1",
            "schema": RECEIPT_FIELDS_V1_SCHEMA,
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    if settings.INFERENCE_BACKEND.strip().lower() == "mock":
        return {"ok": True, "inference_backend": "mock", "vllm_ok": None}

    try:
        models = await vllm.models()
        return {"ok": True, "inference_backend": "vllm", "vllm_ok": True, "models": [m["id"] for m in models.get("data", [])]}
    except Exception as e:
        return {"ok": True, "inference_backend": "vllm", "vllm_ok": False, "error": str(e)}


@app.get("/v1/me")
async def me(principal: ApiPrincipal = Depends(require_scopes([]))):
    return {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))}


@app.post("/v1/debug/dry-run", response_model=DryRunResponse)
async def dry_run_extract(
    req: ExtractReceiptFieldsRequest,
    principal: ApiPrincipal = Depends(require_scopes(["debug:run"])),
) -> DryRunResponse:
    _debug_allowed(principal)

    request_id = _make_request_id(req.request_id)
    prompt_text, temperature, max_tokens = _apply_debug_overrides(principal, req.debug)

    messages = _build_messages(prompt_text, req)
    response_format = _build_response_format()

    return DryRunResponse(
        request_id=request_id,
        model=MODEL_ID,
        inference_backend=settings.INFERENCE_BACKEND.strip().lower(),
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        response_format=response_format,
    )


@app.post("/v1/extract", response_model=ExtractReceiptFieldsResponse)
async def extract_receipt_fields(
    req: ExtractReceiptFieldsRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> ExtractReceiptFieldsResponse:
    request_id = _make_request_id(req.request_id)
    t0 = time.perf_counter()

    prompt_text, temperature, max_tokens = _apply_debug_overrides(principal, req.debug)
    messages = _build_messages(prompt_text, req)
    response_format = _build_response_format()

    backend = settings.INFERENCE_BACKEND.strip().lower()

    try:
        t_inf0 = time.perf_counter()
        if backend == "mock":
            raw = _mock_chat_completions(request_id, principal, req.debug)
        else:
            raw = await vllm.chat_completions(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        t_inf1 = time.perf_counter()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Inference request failed ({backend}): {e}")

    try:
        raw_text = raw["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected {backend} response format.")

    # Parse JSON produced by the model.
    try:
        parsed = json.loads(raw_text)
    except Exception as e:
        artifact_path = _save_artifact(
            request_id,
            {
                "request_id": request_id,
                "model": MODEL_ID,
                "inference_backend": backend,
                "input": req.model_dump(),
                "auth": {"key_id": principal.key_id, "role": principal.role},
                "raw_response": raw,
                "raw_model_text": raw_text,
                "error": f"json_parse_error: {e}",
                "timings_ms": {
                    "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
            },
        )
        if _raw_allowed(principal):
            raise HTTPException(status_code=502, detail=f"Model returned invalid JSON. Artifact: {artifact_path}")
        raise HTTPException(status_code=502, detail="Model returned invalid JSON.")

    errors = [e.message for e in RECEIPT_FIELDS_V1_VALIDATOR.iter_errors(parsed)]
    schema_ok = len(errors) == 0

    _save_artifact(
        request_id,
        {
            "request_id": request_id,
            "model": MODEL_ID,
            "inference_backend": backend,
            "input": req.model_dump(),
            "auth": {"key_id": principal.key_id, "role": principal.role},
            "schema": "receipt_fields_v1",
            "schema_ok": schema_ok,
            "schema_errors": errors,
            "raw_model_text": raw_text,
            "parsed": parsed,
            "raw_response": raw,
            "timings_ms": {
                "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                "total_ms": (time.perf_counter() - t0) * 1000.0,
            },
        },
    )

    raw_out = raw_text if _raw_allowed(principal) else None

    return ExtractReceiptFieldsResponse(
        request_id=request_id,
        model=MODEL_ID,
        schema_ok=schema_ok,
        result=parsed if isinstance(parsed, dict) else {"_parsed": parsed},
        raw_model_text=raw_out,
        timings_ms={
            "inference_ms": (t_inf1 - t_inf0) * 1000.0,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        },
        schema_errors=errors if not schema_ok else None,
    )
