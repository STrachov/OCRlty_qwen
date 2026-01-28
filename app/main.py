import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from app.auth import ApiPrincipal, init_auth_db, require_scopes
from app.settings import settings
from app.vllm_client import VLLMClient

APP_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = APP_ROOT / "schemas"
PROMPTS_DIR = APP_ROOT / "prompts"

# Data root for server-side batch runs.
# By convention in this repo: /workspace/src/data
DATA_ROOT = APP_ROOT.parent.resolve()
DEFAULT_IMAGES_DIR = str((DATA_ROOT / "data").resolve())

ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR).resolve()
MODEL_ID = settings.VLLM_MODEL

# ---------------------------
# Tasks (p.4)
# ---------------------------

TASK_SPECS: Dict[str, Dict[str, str]] = {
    # Existing default task
    "receipt_fields_v1": {
        "schema_file": "receipt_fields_v1.schema.json",
        "schema_name": "default_schema_receipt_fields_v1",
        "prompt_file": "receipt_fields_v1.txt",
        "prompt_name": "default_prompt_receipt_fields_v1",
    },
    # New tasks (add files under app/schemas and app/prompts)
    "payment_v1": {
        "schema_file": "payment_v1.schema.json",
        "schema_name": "payment_v1",
        "prompt_file": "payment_v1.txt",
        "prompt_name": "payment_v1",
    },
    "items_light_v1": {
        "schema_file": "items_light_v1.schema.json",
        "schema_name": "items_light_v1",
        "prompt_file": "items_light_v1.txt",
        "prompt_name": "items_light_v1",
    },
}

# task_id -> runtime payload
_TASKS: Dict[str, Dict[str, Any]] = {}
_TASK_LOAD_ERRORS: Dict[str, str] = {}


def _load_task_or_error(task_id: str) -> None:
    spec = TASK_SPECS[task_id]
    schema_path = SCHEMAS_DIR / spec["schema_file"]
    prompt_path = PROMPTS_DIR / spec["prompt_file"]
    try:
        schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
        prompt_value = prompt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        _TASK_LOAD_ERRORS[task_id] = f"{e.__class__.__name__}: {e}"
        return

    _TASKS[task_id] = {
        "task_id": task_id,
        "schema_value": schema_value,
        "schema_name": spec["schema_name"],
        "prompt_value": prompt_value,
        "prompt_name": spec["prompt_name"],
    }


def _load_tasks() -> None:
    _TASKS.clear()
    _TASK_LOAD_ERRORS.clear()

    for task_id in TASK_SPECS.keys():
        _load_task_or_error(task_id)

    # Default task must exist; otherwise the service isn't usable.
    if "receipt_fields_v1" in _TASK_LOAD_ERRORS:
        raise RuntimeError(
            "Failed to load default task receipt_fields_v1: "
            f"{_TASK_LOAD_ERRORS['receipt_fields_v1']}"
        )


def _get_task(task_id: str) -> Dict[str, Any]:
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is empty.")
    if task_id not in TASK_SPECS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id!r}. Allowed: {sorted(TASK_SPECS.keys())}")
    if task_id in _TASK_LOAD_ERRORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Task files are missing or invalid for task_id={task_id!r}: "
                f"{_TASK_LOAD_ERRORS[task_id]}"
            ),
        )
    return _TASKS[task_id]


# ---------------------------
# App lifecycle
# ---------------------------

vllm = VLLMClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_auth_db()
    _load_tasks()
    yield


app = FastAPI(
    title="Qwen3-VL Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------
# Models
# ---------------------------

class DebugOptions(BaseModel):
    # Any debug usage requires DEBUG_MODE=1 (global env switch).
    prompt_override: Optional[str] = Field(default=None, description="Override the base prompt (DEBUG only).")
    schema_override: Optional[str] = Field(
        default=None,
        description="Override schema (JSON string), or 'none'/'null' to disable schema enforcement (DEBUG only).",
    )
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


class ExtractRequest(BaseModel):
    task_id: str = Field(default="receipt_fields_v1", description="Which extraction task to run.")
    # Provide either image_url OR image_base64.
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_ref: Optional[str] = Field(default=None, description="Optional reference to the source image (e.g., relative path). Stored in artifacts only.")
    mime_type: str = Field(default="image/png", description="Used only with image_base64.")
    request_id: Optional[str] = None

    # Debug block. Presence of this block requires DEBUG_MODE=1 AND debug:run scope.
    debug: Optional[DebugOptions] = None


class ExtractResponse(BaseModel):
    request_id: str
    task_id: str
    model: str

    schema_ok: bool
    result: Dict[str, Any]
    timings_ms: Dict[str, float]
    schema_errors: Optional[list[str]] = None

    # Returned only when DEBUG_MODE=1 and caller has debug:read_raw (or AUTH is disabled).
    raw_model_text: Optional[str] = None


class ArtifactListItem(BaseModel):
    request_id: str
    date: str
    task_id: Optional[str] = None
    schema_ok: Optional[bool] = None


class ArtifactListResponse(BaseModel):
    items: List[ArtifactListItem]

class BatchExtractRequest(BaseModel):
    task_id: str = Field(default="receipt_fields_v1", description="Which extraction task to run for all images.")
    images_dir: str = Field(default=DEFAULT_IMAGES_DIR, description="Server-side directory with images (must be under /workspace/src).")
    glob: str = Field(default="**/*", description="Glob pattern relative to images_dir.")
    exts: List[str] = Field(default_factory=lambda: ["png", "jpg", "jpeg", "webp"], description="Allowed file extensions (without dot).")
    limit: Optional[int] = Field(default=None, ge=1, description="Optional max number of images to process.")
    concurrency: int = Field(default=4, ge=1, le=32, description="Number of concurrent requests within this process.")
    run_id: Optional[str] = Field(default=None, description="Optional run id (used to build per-image request_id).")


class BatchExtractResponse(BaseModel):
    run_id: str
    task_id: str
    images_dir: str

    total: int
    ok: int
    failed: int
    schema_ok: int
    schema_failed: int

    timings_ms: Dict[str, float]
    batch_artifact_path: str



# ---------------------------
# Helpers
# ---------------------------

_REQUEST_ID_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _make_request_id(user_request_id: Optional[str]) -> str:
    return user_request_id or uuid.uuid4().hex


def _validate_request_id_for_fs(request_id: str) -> None:
    if not _REQUEST_ID_SAFE.match(request_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid request_id (allowed: A-Za-z0-9_.-, max 128 chars).",
        )


def _ensure_artifacts_dir() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_artifact(request_id: str, payload: Dict[str, Any]) -> str:
    _validate_request_id_for_fs(request_id)
    out_dir = _ensure_artifacts_dir()
    path = out_dir / f"{request_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)



def _ensure_batch_artifacts_dir() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_dir = ARTIFACTS_DIR / "batches" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_batch_artifact(run_id: str, payload: Dict[str, Any]) -> str:
    _validate_request_id_for_fs(run_id)
    out_dir = _ensure_batch_artifacts_dir()
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _validate_under_data_root(p: Path) -> Path:
    try:
        p_rel = p.resolve().relative_to(DATA_ROOT)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Path must be under data root {str(DATA_ROOT)!r}. Got: {str(p)!r}",
        )
    return DATA_ROOT / p_rel


def _list_image_files(images_dir: Path, glob_pattern: str, exts: List[str], limit: Optional[int]) -> List[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"images_dir does not exist or is not a directory: {str(images_dir)!r}")

    allowed = {("." + e.lower().lstrip(".")) for e in (exts or [])}
    files: List[Path] = []
    for p in images_dir.glob(glob_pattern or "**/*"):
        if not p.is_file():
            continue
        if allowed and p.suffix.lower() not in allowed:
            continue
        files.append(p)

    files.sort(key=lambda x: str(x))
    if limit is not None:
        files = files[: int(limit)]
    return files


def _default_run_id() -> str:
    # Must match request_id regex and remain short.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]


def _make_batch_request_id(run_id: str, rel_path: str) -> str:
    # Keep it stable and short: <run_id>__<base>_<hash8>
    base = os.path.splitext(os.path.basename(rel_path))[0]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:40] or "img"
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
    req_id = f"{run_id}__{base}_{h}"
    if len(req_id) > 128:
        req_id = req_id[:128]
    _validate_request_id_for_fs(req_id)
    return req_id


def _guess_mime_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "image/png"


def _image_content(image_url: Optional[str], image_base64: Optional[str], mime_type: str) -> Dict[str, Any]:
    if image_url and image_base64:
        raise HTTPException(status_code=400, detail="Provide either image_url or image_base64 (not both).")
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    if image_base64:
        data_url = f"data:{mime_type};base64,{image_base64}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    raise HTTPException(status_code=400, detail="Provide either image_url or image_base64.")



def _sanitize_input_for_artifact(req: "ExtractRequest") -> Dict[str, Any]:
    # Avoid storing large blobs (base64) in artifacts.
    return {
        "task_id": req.task_id,
        "user_request_id": req.request_id,
        "image_ref": req.image_ref,
        "image_url": req.image_url,
        "has_image_base64": bool(req.image_base64),
        "image_base64_chars": len(req.image_base64) if req.image_base64 else 0,
        "mime_type": req.mime_type,
        "has_debug": bool(req.debug),
    }

def _debug_enabled_or_403() -> None:
    if not settings.DEBUG_MODE:
        raise HTTPException(status_code=403, detail="Debug features are disabled (DEBUG_MODE=0).")


def _debug_allowed(principal: ApiPrincipal) -> None:
    _debug_enabled_or_403()
    if settings.AUTH_ENABLED and ("debug:run" not in principal.scopes):
        raise HTTPException(status_code=403, detail="Missing required scope: debug:run")


def _require_debug_read_raw(principal: ApiPrincipal) -> None:
    _debug_enabled_or_403()
    if settings.AUTH_ENABLED and ("debug:read_raw" not in principal.scopes):
        raise HTTPException(status_code=403, detail="Missing required scope: debug:read_raw")


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
    *,
    base_prompt_text: str,
    base_schema_json: dict[str, Any],
) -> Tuple[str, float, int, Union[dict[str, Any], str, None]]:
    prompt_text = base_prompt_text
    schema_json: Union[dict[str, Any], str, None] = base_schema_json
    temperature = 0.0
    max_tokens = 128

    if debug is None:
        return prompt_text, temperature, max_tokens, schema_json

    _debug_allowed(principal)

    if debug.prompt_override is not None:
        p = debug.prompt_override.strip()
        if len(p) > int(settings.MAX_PROMPT_CHARS):
            raise HTTPException(status_code=400, detail=f"prompt_override too long (>{settings.MAX_PROMPT_CHARS} chars).")
        if p:
            prompt_text = p

    if debug.schema_override is not None:
        s = debug.schema_override.strip()
        if s:
            schema_json = s

    if debug.temperature is not None:
        temperature = float(debug.temperature)

    if debug.max_tokens is not None:
        max_tokens = int(debug.max_tokens)

    max_tokens = _clamp_max_tokens(max_tokens)
    return prompt_text, temperature, max_tokens, schema_json


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
    schema_dict: Optional[dict[str, Any]],
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
        if schema_dict is None:
            content = json.dumps({"mock": True, "schema_used": False}, ensure_ascii=False)
        else:
            sample = _sample_from_schema(schema_dict, schema_dict)
            content = json.dumps(sample, ensure_ascii=False)

    return {"choices": [{"message": {"content": content}}]}


def _build_messages(prompt_text: str, req: ExtractRequest) -> Any:
    return [
        {
            "role": "user",
            "content": [
                _image_content(req.image_url, req.image_base64, req.mime_type),
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _normalize_schema(schema: Union[dict[str, Any], str, None], default_schema: dict[str, Any]) -> Optional[dict[str, Any]]:
    if schema is None:
        return default_schema

    if isinstance(schema, dict):
        return schema

    if isinstance(schema, str):
        s = schema.strip().lower()
        if s in {"none", "null"}:
            return None

        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"schema is a string but not valid JSON: {schema!r}")

        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail=f"schema JSON must decode to dict, got {type(parsed).__name__}")

        return parsed

    raise HTTPException(
        status_code=400,
        detail=f"schema must be dict | json-str | 'none' | None, got {type(schema).__name__}",
    )


def _build_response_format(schema_dict: dict[str, Any], schema_name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_dict,
        },
    }


def _make_validator_or_raise(schema_dict: dict[str, Any], *, is_custom_schema: bool) -> Draft202012Validator:
    try:
        Draft202012Validator.check_schema(schema_dict)
    except SchemaError as e:
        msg = getattr(e, "message", str(e))
        if is_custom_schema:
            raise HTTPException(status_code=400, detail=f"Invalid JSON schema in schema_override: {msg}")
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: invalid default schema: {msg}")

    try:
        return Draft202012Validator(schema_dict)
    except Exception as e:
        if is_custom_schema:
            raise HTTPException(status_code=400, detail=f"Invalid JSON schema in schema_override: {e}")
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: invalid default schema: {e}")


# ---------------------------
# Artifacts 
# ---------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iter_artifact_days_desc() -> List[str]:
    if not ARTIFACTS_DIR.exists():
        return []
    days = []
    for p in ARTIFACTS_DIR.iterdir():
        if p.is_dir() and _DATE_RE.match(p.name):
            days.append(p.name)
    days.sort(reverse=True)
    return days


def _find_artifact_path(request_id: str, *, date: Optional[str] = None, max_days: int = 30) -> Optional[Path]:
    _validate_request_id_for_fs(request_id)

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")
        p = ARTIFACTS_DIR / date / f"{request_id}.json"
        return p if p.exists() else None

    days = _iter_artifact_days_desc()[: max(1, int(max_days))]
    for d in days:
        p = ARTIFACTS_DIR / d / f"{request_id}.json"
        if p.exists():
            return p
    return None


def _artifact_date_from_path(p: Path) -> Optional[str]:
    # /.../artifacts/YYYY-MM-DD/<id>.json
    try:
        return p.parent.name
    except Exception:
        return None


def _read_artifact_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read artifact JSON: {e}")


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    backend = settings.INFERENCE_BACKEND.strip().lower()
    out: Dict[str, Any] = {"ok": True, "inference_backend": backend}

    # show tasks status
    out["tasks_available"] = sorted(list(_TASKS.keys()))
    out["tasks_missing"] = dict(_TASK_LOAD_ERRORS)

    if backend == "mock":
        out["vllm_ok"] = None
        return out

    try:
        models = await vllm.models()
        out["vllm_ok"] = True
        out["models"] = [m.get("id") for m in models.get("data", [])]
        return out
    except Exception as e:
        out["vllm_ok"] = False
        out["error"] = str(e)
        return out


@app.get("/v1/me")
async def me(principal: ApiPrincipal = Depends(require_scopes([]))):
    return {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))}


# --- p.2: debug artifact viewing ---

@app.get("/v1/debug/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    date: Optional[str] = Query(default=None, description="Filter by day (YYYY-MM-DD). If omitted, returns most recent items across days."),
    limit: int = Query(default=50, ge=1, le=500),
    principal: ApiPrincipal = Depends(require_scopes(["debug:read_raw"])),
) -> ArtifactListResponse:
    _require_debug_read_raw(principal)

    items: List[ArtifactListItem] = []

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")
        day_dir = ARTIFACTS_DIR / date
        if not day_dir.exists():
            return ArtifactListResponse(items=[])

        files = sorted(day_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:limit]:
            rid = p.stem
            meta = ArtifactListItem(request_id=rid, date=date)
            # best-effort parse minimal fields
            try:
                data = _read_artifact_json(p)
                meta.task_id = data.get("task_id")
                meta.schema_ok = data.get("schema_ok")
            except Exception:
                pass
            items.append(meta)

        return ArtifactListResponse(items=items)

    # no date: walk recent day folders and collect
    for d in _iter_artifact_days_desc():
        day_dir = ARTIFACTS_DIR / d
        files = sorted(day_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            rid = p.stem
            meta = ArtifactListItem(request_id=rid, date=d)
            try:
                data = _read_artifact_json(p)
                meta.task_id = data.get("task_id")
                meta.schema_ok = data.get("schema_ok")
            except Exception:
                pass
            items.append(meta)
            if len(items) >= limit:
                return ArtifactListResponse(items=items)

    return ArtifactListResponse(items=items)


@app.get("/v1/debug/artifacts/{request_id}")
async def read_artifact(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
    principal: ApiPrincipal = Depends(require_scopes(["debug:read_raw"])),
) -> Dict[str, Any]:
    _require_debug_read_raw(principal)

    p = _find_artifact_path(request_id, date=date)
    if p is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    data = _read_artifact_json(p)
    # include a tiny bit of metadata without absolute paths
    return {
        "request_id": request_id,
        "date": _artifact_date_from_path(p),
        "artifact": data,
    }


@app.get("/v1/debug/artifacts/{request_id}/raw")
async def read_artifact_raw_text(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
    principal: ApiPrincipal = Depends(require_scopes(["debug:read_raw"])),
) -> Dict[str, Any]:
    _require_debug_read_raw(principal)

    p = _find_artifact_path(request_id, date=date)
    if p is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    data = _read_artifact_json(p)
    return {
        "request_id": request_id,
        "date": _artifact_date_from_path(p),
        "raw_model_text": data.get("raw_model_text"),
    }


# --- p.4: extract with task_id ---

@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    req: ExtractRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> ExtractResponse:
    task = _get_task(req.task_id)

    request_id = _make_request_id(req.request_id)
    _validate_request_id_for_fs(request_id)

    # image_url is allowed only as a DEBUG feature (DEBUG_MODE=1 and debug:run scope).
    if req.image_url:
        _debug_allowed(principal)

    t0 = time.perf_counter()

    prompt_text, temperature, max_tokens, schema_json = _apply_debug_overrides(
        principal,
        req.debug,
        base_prompt_text=task["prompt_value"],
        base_schema_json=task["schema_value"],
    )

    messages = _build_messages(prompt_text, req)

    schema_dict = _normalize_schema(schema_json, default_schema=task["schema_value"])
    is_custom_schema = bool(req.debug and req.debug.schema_override and req.debug.schema_override.strip())
    schema_name = "custom_schema" if is_custom_schema else task["schema_name"]
    response_format = _build_response_format(schema_dict, schema_name=schema_name) if schema_dict is not None else None

    backend = settings.INFERENCE_BACKEND.strip().lower()

    try:
        t_inf0 = time.perf_counter()
        if backend == "mock":
            raw = _mock_chat_completions(request_id, principal, req.debug, schema_dict=schema_dict)
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
                "task_id": req.task_id,
                "model": MODEL_ID,
                "inference_backend": backend,
                "input": _sanitize_input_for_artifact(req),
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

    if schema_dict is None:
        schema_ok = True
        errors: list[str] = []
    else:
        schema_json_validator = _make_validator_or_raise(schema_dict, is_custom_schema=is_custom_schema)
        errors = [e.message for e in schema_json_validator.iter_errors(parsed)]
        schema_ok = len(errors) == 0

    _save_artifact(
        request_id,
        {
            "request_id": request_id,
            "task_id": req.task_id,
            "model": MODEL_ID,
            "inference_backend": backend,
            "input": _sanitize_input_for_artifact(req),
            "auth": {"key_id": principal.key_id, "role": principal.role},
            "schema": None if schema_dict is None else json.dumps(schema_dict, ensure_ascii=False),
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

    return ExtractResponse(
        request_id=request_id,
        task_id=req.task_id,
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


# --- Batch extract (server-side folder) ---

@app.post("/v1/batch_extract", response_model=BatchExtractResponse)
async def batch_extract(
    req: BatchExtractRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> BatchExtractResponse:
    # Validate task files early.
    _get_task(req.task_id)

    run_id = (req.run_id or _default_run_id()).strip()
    _validate_request_id_for_fs(run_id)

    images_dir = _validate_under_data_root(Path(req.images_dir).expanduser())
    files = _list_image_files(images_dir, req.glob, req.exts, req.limit)

    t0 = time.perf_counter()

    sem = asyncio.Semaphore(int(req.concurrency))
    items: List[Dict[str, Any]] = []

    async def _process_one(p: Path) -> Dict[str, Any]:
        rel_path = str(p.relative_to(images_dir)).replace("\\", "/")
        request_id = _make_batch_request_id(run_id, rel_path)

        async with sem:
            try:
                data = await asyncio.to_thread(p.read_bytes)
                b64 = base64.b64encode(data).decode("ascii")
                mime = _guess_mime_type(p)

                sub_req = ExtractRequest(
                    task_id=req.task_id,
                    image_base64=b64,
                    mime_type=mime,
                    request_id=request_id,
                    image_ref=rel_path,
                    debug=None,
                )

                resp = await extract(sub_req, principal)  # reuse single-image logic
                return {
                    "file": rel_path,
                    "request_id": resp.request_id,
                    "ok": True,
                    "schema_ok": resp.schema_ok,
                    "timings_ms": resp.timings_ms,
                }
            except HTTPException as he:
                return {
                    "file": rel_path,
                    "request_id": request_id,
                    "ok": False,
                    "schema_ok": False,
                    "error": f"HTTP {he.status_code}: {he.detail}",
                }
            except Exception as e:
                return {
                    "file": rel_path,
                    "request_id": request_id,
                    "ok": False,
                    "schema_ok": False,
                    "error": f"{e.__class__.__name__}: {e}",
                }

    # Run
    results = await asyncio.gather(*[_process_one(p) for p in files])
    items.extend(results)

    total = len(items)
    ok = sum(1 for it in items if it.get("ok"))
    failed = total - ok
    schema_ok = sum(1 for it in items if it.get("ok") and it.get("schema_ok"))
    schema_failed = sum(1 for it in items if it.get("ok") and (not it.get("schema_ok")))

    t1 = time.perf_counter()

    batch_artifact_path = _save_batch_artifact(
        run_id,
        {
            "run_id": run_id,
            "task_id": req.task_id,
            "images_dir": str(images_dir),
            "glob": req.glob,
            "exts": req.exts,
            "limit": req.limit,
            "concurrency": req.concurrency,
            "model": MODEL_ID,
            "inference_backend": settings.INFERENCE_BACKEND.strip().lower(),
            "auth": {"key_id": principal.key_id, "role": principal.role},
            "summary": {
                "total": total,
                "ok": ok,
                "failed": failed,
                "schema_ok": schema_ok,
                "schema_failed": schema_failed,
            },
            "timings_ms": {
                "total_ms": (t1 - t0) * 1000.0,
            },
            "items": items,
        },
    )

    return BatchExtractResponse(
        run_id=run_id,
        task_id=req.task_id,
        images_dir=str(images_dir),
        total=total,
        ok=ok,
        failed=failed,
        schema_ok=schema_ok,
        schema_failed=schema_failed,
        timings_ms={"total_ms": (t1 - t0) * 1000.0},
        batch_artifact_path=batch_artifact_path,
    )
