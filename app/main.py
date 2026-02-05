import asyncio
import base64
import math
from io import BytesIO

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None
import hashlib
import json
import mimetypes
import os
import re
import unicodedata
import time
import uuid
import tarfile
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
from collections import defaultdict

import boto3
import botocore
from botocore.config import Config as BotoConfig

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
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
# Artifact storage (S3/R2 or local filesystem)
# ---------------------------
# Enable S3 mode by setting S3_BUCKET. For Cloudflare R2, also set:
#   S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
#   S3_REGION=auto
# and provide AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (R2 S3 API tokens).
_S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
_S3_ENDPOINT_URL = (os.getenv("S3_ENDPOINT_URL", "").strip() or None)
_S3_REGION = os.getenv("S3_REGION", "auto").strip()  # R2 uses 'auto' (required by SDK but not used).
_S3_PREFIX = (os.getenv("S3_PREFIX", os.getenv("S3_ARTIFACTS_PREFIX", "ocrlty"))).strip().strip("/")
_S3_ALLOW_OVERWRITE = os.getenv("S3_ALLOW_OVERWRITE", "0").lower() in ("1", "true", "yes", "y")
_S3_PRESIGN_TTL_S = int(os.getenv("S3_PRESIGN_TTL_S", "3600"))
_S3_FORCE_PATH_STYLE = os.getenv("S3_FORCE_PATH_STYLE", "0").lower() in ("1", "true", "yes", "y")

_s3_client_singleton = None


def _s3_enabled() -> bool:
    return bool(_S3_BUCKET)


def _s3_client():
    global _s3_client_singleton
    if _s3_client_singleton is not None:
        return _s3_client_singleton

    # Credentials: prefer standard AWS env vars, but also accept S3_* variants.
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID") or ""
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY") or ""

    # Let boto3 resolve creds if not explicitly provided.
    kwargs: Dict[str, Any] = {
        "service_name": "s3",
        "region_name": _S3_REGION or None,
        "endpoint_url": _S3_ENDPOINT_URL,
        "config": BotoConfig(
            retries={"max_attempts": 8, "mode": "standard"},
            s3={"addressing_style": ("path" if _S3_FORCE_PATH_STYLE else "virtual")},
        ),
    }
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key

    _s3_client_singleton = boto3.client(**kwargs)
    return _s3_client_singleton


def _s3_key(rel: str) -> str:
    rel = (rel or "").lstrip("/")
    if _S3_PREFIX:
        return f"{_S3_PREFIX}/{rel}" if rel else _S3_PREFIX
    return rel


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _s3_is_retryable_client_error(e: Exception) -> bool:
    if not isinstance(e, botocore.exceptions.ClientError):
        return False
    code = (e.response or {}).get("Error", {}).get("Code", "") or ""
    status = int((e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
    # Retry on transient-ish errors.
    if status >= 500:
        return True
    if code in {"RequestTimeout", "Throttling", "ThrottlingException", "SlowDown", "InternalError"}:
        return True
    return False


def _s3_call_with_retries(fn, *, op: str, max_attempts: int = 5) -> Any:
    # boto3 has built-in retries, but we still wrap it to handle edge network cases.
    delay_s = 0.25
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            retryable = _s3_is_retryable_client_error(e) or isinstance(
                e,
                (
                    botocore.exceptions.EndpointConnectionError,
                    botocore.exceptions.ConnectionClosedError,
                    botocore.exceptions.ReadTimeoutError,
                    botocore.exceptions.ConnectTimeoutError,
                ),
            )
            if not retryable or attempt == max_attempts:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2.0, 2.0)
    raise last_err  # pragma: no cover


def _s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def _do():
        kwargs: Dict[str, Any] = {
            "Bucket": _S3_BUCKET,
            "Key": key,
            "Body": body,
            "ContentType": "application/json; charset=utf-8",
            "CacheControl": "no-store",
        }
        if not _S3_ALLOW_OVERWRITE:
            # Avoid accidental overwrites if request_id/run_id is reused.
            kwargs["IfNoneMatch"] = "*"
        return _s3_client().put_object(**kwargs)

    try:
        _s3_call_with_retries(_do, op="put_object")
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        # If overwrites are disallowed and the object already exists, treat as success.
        if code in {"PreconditionFailed"} and not _S3_ALLOW_OVERWRITE:
            return
        raise


def _s3_get_text(key: str) -> str:
    def _do():
        return _s3_client().get_object(Bucket=_S3_BUCKET, Key=key)

    try:
        obj = _s3_call_with_retries(_do, op="get_object")
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"NoSuchKey", "404"}:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        raise HTTPException(status_code=502, detail=f"S3 get_object failed: {code}")

    body = obj["Body"].read()
    return body.decode("utf-8", errors="replace")


def _s3_get_json(key: str) -> Dict[str, Any]:
    try:
        return json.loads(_s3_get_text(key))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse artifact JSON: {e}")


def _s3_exists(key: str) -> bool:
    def _do():
        return _s3_client().head_object(Bucket=_S3_BUCKET, Key=key)

    try:
        _s3_call_with_retries(_do, op="head_object", max_attempts=3)
        return True
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"NoSuchKey", "404"}:
            return False
        status = int((e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
        if status == 404:
            return False
        raise


def _s3_list_common_prefixes(prefix: str) -> List[str]:
    # Lists immediate children prefixes under prefix (Delimiter="/").
    out: List[str] = []
    token: Optional[str] = None
    while True:
        def _do():
            kwargs: Dict[str, Any] = {"Bucket": _S3_BUCKET, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kwargs["ContinuationToken"] = token
            return _s3_client().list_objects_v2(**kwargs)

        resp = _s3_call_with_retries(_do, op="list_objects_v2")
        for cp in resp.get("CommonPrefixes") or []:
            pfx = cp.get("Prefix") or ""
            if pfx:
                out.append(pfx)
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out


def _s3_list_objects(prefix: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        def _do():
            kwargs: Dict[str, Any] = {"Bucket": _S3_BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            return _s3_client().list_objects_v2(**kwargs)

        resp = _s3_call_with_retries(_do, op="list_objects_v2")
        for it in resp.get("Contents") or []:
            out.append(it)
            if limit is not None and len(out) >= limit:
                return out
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out


def _artifact_key(kind: str, day: str, name: str) -> str:
    return _s3_key(f"{kind}/{day}/{name}")


def _extract_artifact_ref(request_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if _s3_enabled():
        return _artifact_key("extracts", d, f"{request_id}.json")
    return (ARTIFACTS_DIR / "extracts" / d / f"{request_id}.json").resolve()


def _batch_artifact_ref(run_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if _s3_enabled():
        return _artifact_key("batches", d, f"{run_id}.json")
    return (ARTIFACTS_DIR / "batches" / d / f"{run_id}.json").resolve()


def _eval_artifact_ref(eval_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if _s3_enabled():
        return _artifact_key("evals", d, f"{eval_id}.json")
    return (ARTIFACTS_DIR / "evals" / d / f"{eval_id}.json").resolve()

# ---------------------------
# Context-length safety / smart image retry
# ---------------------------
# We don't have exact multimodal tokenization locally. Instead, on "decoder prompt too long" errors,
# we parse the server-provided token counts and compute a one-shot downscale factor for the image.
_CTX_SAFETY_TOKENS = 256          # keep some slack for chat template / special tokens
_SCALE_SAFETY = 0.90              # extra shrink for safety
_MIN_SCALE = 0.20                 # do not shrink below this in one retry
_MAX_SCALE = 0.95                 # avoid pointless re-encode
_QWEN_ALIGN = 32                  # Qwen-VL family typically aligns vision sizes to multiples of 32
_TEXT_CHARS_PER_TOKEN = 3.0       # conservative heuristic for "JSON-y" text


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

    # Populated when we had to downscale the image due to context-length constraints.
    image_resize: Optional[Dict[str, Any]] = None

    # System-level errors/retries recorded during processing (empty for a clean successful run).
    error_history: List[Dict[str, Any]] = Field(default_factory=list)

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

    gt_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to GT JSON. If provided, the server will run batch eval vs GT after batch_extract. "
            "For safety this requires DEBUG_MODE=1 and scope debug:run. Path must be under DATA_ROOT or ARTIFACTS_DIR."
        ),
    )
    gt_image_key: str = Field(
        default="image",
        description="Key in each GT record that contains image path/name (basename is used). Fallback: image_file.",
    )


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

    eval: Optional[Dict[str, Any]] = Field(default=None, description="Optional eval result (if gt_path was provided).")
    



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
    # Local filesystem mode only.
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / "extracts" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_artifact(request_id: str, payload: Dict[str, Any]) -> str:
    _validate_request_id_for_fs(request_id)
    day = _utc_day()

    if _s3_enabled():
        key = str(_extract_artifact_ref(request_id, day=day))
        try:
            _s3_put_json(key, payload)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to write artifact to S3: {e}")
        return key

    out_dir = _ensure_artifacts_dir()
    path = out_dir / f"{request_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _to_artifact_rel(p: Union[str, Path]) -> str:
    """
    For local mode: path relative to ARTIFACTS_DIR.
    For S3 mode: key relative to S3_PREFIX (so logs/UI stay readable).
    """
    if p is None:
        return ""
    s = str(p)

    if _s3_enabled():
        if _S3_PREFIX and s.startswith(_S3_PREFIX + "/"):
            return s[len(_S3_PREFIX) + 1 :]
        return s

    try:
        pp = Path(s).resolve()
        return str(pp.relative_to(ARTIFACTS_DIR))
    except Exception:
        return s


def _ensure_batch_artifacts_dir() -> Path:
    # Local filesystem mode only.
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / "batches" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_batch_artifact(run_id: str, payload: Dict[str, Any]) -> str:
    _validate_request_id_for_fs(run_id)
    day = _utc_day()

    if _s3_enabled():
        key = str(_batch_artifact_ref(run_id, day=day))
        try:
            _s3_put_json(key, payload)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to write batch artifact to S3: {e}")
        return key

    out_dir = _ensure_batch_artifacts_dir()
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _ensure_eval_artifacts_dir() -> Path:
    # Local filesystem mode only.
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / "evals" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_eval_artifact(payload: Dict[str, Any]) -> str:
    eval_id = str(payload.get("eval_id") or "")
    _validate_request_id_for_fs(eval_id)
    day = _utc_day()

    if _s3_enabled():
        key = str(_eval_artifact_ref(eval_id, day=day))
        try:
            _s3_put_json(key, payload)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to write eval artifact to S3: {e}")
        return key

    out_dir = _ensure_eval_artifacts_dir()
    path = out_dir / f"{eval_id}.json"
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



def _sanitize_input_for_artifact(req: "ExtractRequest", image_resize: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Avoid storing large blobs (base64) in artifacts.
    return {
        "task_id": req.task_id,
        "user_request_id": req.request_id,
        "image_ref": req.image_ref,
        "image_url": req.image_url,
        "has_image_base64": bool(req.image_base64),
        "image_base64_chars": len(req.image_base64) if req.image_base64 else 0,
        "mime_type": req.mime_type,
        "image_resize": image_resize,
        "has_debug": bool(req.debug),
    }

def _debug_enabled() -> None:
    if not settings.DEBUG_MODE:
        raise HTTPException(status_code=403, detail="Debug features are disabled (DEBUG_MODE=0).")


def _debug_allowed(principal: ApiPrincipal) -> None:
    _debug_enabled()
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


# ---------------------------
# Smart retry for context-length errors (decoder prompt too long)
# ---------------------------
# Fragment of the error: The decoder prompt (length 9254) is longer than the maximum model length of 4096. Make sure that `max_model_len` is no smaller ...
_DECODER_TOO_LONG_RE = re.compile(
    r"(?:decoder\s+)?prompt\s*\(length\s+(\d+)\)\s+is\s+longer\s+than\s+the\s+maximum\s+model\s+length\s+of\s+(\d+)",
    re.IGNORECASE,
)


def _parse_decoder_prompt_too_long(err: str) -> Optional[Tuple[int, int]]:
    """Return (t_resp, t_max) if error looks like a context-length error."""
    if not err:
        return None
    m = _DECODER_TOO_LONG_RE.search(err)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def _estimate_prompt_tokens(prompt_text: str, schema_dict: Optional[dict[str, Any]]) -> int:
    """Heuristic token count for the *text* part of the request (prompt + schema).

    We cannot reproduce exact multimodal tokenization locally. This estimate is used only to compute
    an image downscale factor after the server returns an exact total length (t_resp).
    """
    schema_s = ""
    if schema_dict is not None:
        try:
            schema_s = json.dumps(schema_dict, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            schema_s = str(schema_dict)

    chars = len(prompt_text) + len(schema_s)
    # Add slack for chat template, role wrappers, and structured output glue.
    chars += 1000
    return int(math.ceil(chars / _TEXT_CHARS_PER_TOKEN))


def _align_down(x: int, m: int) -> int:
    if x <= 0:
        return m
    return max(m, (x // m) * m)


def _resize_base64_image_by_scale(image_base64: str, mime_type: str, scale: float) -> Tuple[str, Dict[str, Any]]:
    if Image is None:
        raise RuntimeError("PIL (Pillow) is required for image resizing but is not available.")
    data = base64.b64decode(image_base64)
    im = Image.open(BytesIO(data))
    im.load()
    w0, h0 = im.size

    # Compute new size (align to 32 for Qwen-VL style models).
    new_w = _align_down(int(w0 * scale), _QWEN_ALIGN)
    new_h = _align_down(int(h0 * scale), _QWEN_ALIGN)

    # Avoid degenerate sizes.
    new_w = max(_QWEN_ALIGN, new_w)
    new_h = max(_QWEN_ALIGN, new_h)

    # If scaling doesn't change anything, return original.
    if new_w >= w0 and new_h >= h0:
        return image_base64, {
            "applied": False,
            "reason": "scale_noop",
            "scale": float(scale),
            "orig_w": int(w0),
            "orig_h": int(h0),
            "new_w": int(w0),
            "new_h": int(h0),
        }

    im = im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)

    out = BytesIO()
    mt = (mime_type or "").lower().strip()

    # Preserve PNG when explicitly requested; otherwise JPEG is fine for receipts.
    if mt == "image/png":
        im.save(out, format="PNG", optimize=True)
        out_mime = "image/png"
    else:
        im.save(out, format="JPEG", quality=92, optimize=True, progressive=True)
        out_mime = "image/jpeg"

    b64_new = base64.b64encode(out.getvalue()).decode("ascii")
    return b64_new, {
        "applied": True,
        "reason": "context_len_retry",
        "scale": float(scale),
        "orig_w": int(w0),
        "orig_h": int(h0),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "orig_b64_chars": int(len(image_base64)),
        "new_b64_chars": int(len(b64_new)),
        "sent_mime_type": out_mime,
    }


def _compute_retry_scale(t_resp: int, t_max: int, t_prompt_est: int) -> Optional[float]:
    """Compute linear image downscale factor from server-provided lengths.

    Let:
      t_resp = total decoder prompt length from the server error (text + image tokens)
      t_prompt_est = estimated text-only tokens (prompt + schema + wrappers)
      t_max = server max context length

    We want: t_prompt_est + t_img_new <= t_max - safety
    and assume image tokens scale ~ area ~ (scale^2).
    """
    budget = int(t_max) - int(_CTX_SAFETY_TOKENS)
    if t_prompt_est >= budget:
        return None

    t_img_now = max(1, int(t_resp) - int(t_prompt_est))
    t_img_budget = max(1, int(budget) - int(t_prompt_est))

    # token reduction ratio needed
    r_tokens = t_img_budget / float(t_img_now)
    # linear scale ~ sqrt(token_ratio)
    scale = math.sqrt(r_tokens) * float(_SCALE_SAFETY)
    scale = max(float(_MIN_SCALE), min(float(_MAX_SCALE), float(scale)))
    return scale


def _now_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace('+00:00', 'Z')


def _mk_err(stage: str, message: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "ts": _now_iso_ms(),
        "stage": stage,
        "message": message,
    }
    if extra:
        d["extra"] = extra
    return d


def _try_load_error_history_for_request(request_id: str) -> Optional[list[Dict[str, Any]]]:
    art_p = _find_artifact_path(request_id, max_days=30)
    if art_p is None:
        return None
    try:
        obj = json.loads(Path(art_p).read_text(encoding="utf-8"))
        eh = obj.get("error_history")
        return eh if isinstance(eh, list) else None
    except Exception:
        return None


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
            "strict": True,
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
    if _s3_enabled():
        base = _s3_key("extracts/")
        days: List[str] = []
        for pfx in _s3_list_common_prefixes(base):
            m = re.search(r"/extracts/(\d{4}-\d{2}-\d{2})/$", pfx)
            if m:
                days.append(m.group(1))
        days.sort(reverse=True)
        return days

    base = ARTIFACTS_DIR / "extracts"
    if not base.exists():
        return []
    days = []
    for p in base.iterdir():
        if p.is_dir() and _DATE_RE.match(p.name):
            days.append(p.name)
    days.sort(reverse=True)
    return days


def _find_artifact_path(request_id: str, *, date: Optional[str] = None, max_days: int = 30) -> Optional[Union[Path, str]]:
    _validate_request_id_for_fs(request_id)

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")
        ref = _extract_artifact_ref(request_id, day=date)
        if _s3_enabled():
            return str(ref) if _s3_exists(str(ref)) else None
        p = Path(ref)
        return p if p.exists() else None

    days = _iter_artifact_days_desc()[: max(1, int(max_days))]
    for d in days:
        ref = _extract_artifact_ref(request_id, day=d)
        if _s3_enabled():
            if _s3_exists(str(ref)):
                return str(ref)
        else:
            p = Path(ref)
            if p.exists():
                return p
    return None


def _artifact_date_from_path(p: Union[str, Path]) -> Optional[str]:
    if isinstance(p, Path):
        try:
            return p.parent.name
        except Exception:
            return None

    s = str(p)
    m = re.search(r"/(extracts|batches|evals)/(\d{4}-\d{2}-\d{2})/", s)
    return m.group(2) if m else None


def _read_artifact_json(p: Union[str, Path]) -> Dict[str, Any]:
    try:
        if _s3_enabled() and not isinstance(p, Path):
            return _s3_get_json(str(p))
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read artifact JSON: {e}")



# ---------------------------
# Batch artifacts + evaluation helpers
# ---------------------------

def _iter_batch_days_desc() -> List[str]:
    if _s3_enabled():
        base = _s3_key("batches/")
        days: List[str] = []
        for pfx in _s3_list_common_prefixes(base):
            m = re.search(r"/batches/(\d{4}-\d{2}-\d{2})/$", pfx)
            if m:
                days.append(m.group(1))
        days.sort(reverse=True)
        return days

    base = ARTIFACTS_DIR / "batches"
    if not base.exists():
        return []
    days: List[str] = []
    for p in base.iterdir():
        if p.is_dir() and _DATE_RE.match(p.name):
            days.append(p.name)
    days.sort(reverse=True)
    return days


def _find_batch_artifact_path(run_id: str, *, date: Optional[str] = None, max_days: int = 30) -> Optional[Union[Path, str]]:
    _validate_request_id_for_fs(run_id)

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="batch_date must be in YYYY-MM-DD format.")
        ref = _batch_artifact_ref(run_id, day=date)
        if _s3_enabled():
            return str(ref) if _s3_exists(str(ref)) else None
        p = Path(ref)
        return p if p.exists() else None

    days = _iter_batch_days_desc()[: max(1, int(max_days))]
    for d in days:
        ref = _batch_artifact_ref(run_id, day=d)
        if _s3_enabled():
            if _s3_exists(str(ref)):
                return str(ref)
        else:
            p = Path(ref)
            if p.exists():
                return p
    return None


def _read_json_file(p: Union[str, Path]) -> Any:
    try:
        if _s3_enabled() and not isinstance(p, Path):
            return json.loads(_s3_get_text(str(p)))
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read JSON file {str(p)!r}: {e}")


def _validate_under_any_root(p: Path, roots: List[Path]) -> Path:
    pp = p.expanduser().resolve()
    for r in roots:
        try:
            rel = pp.relative_to(r.resolve())
            return r.resolve() / rel
        except Exception:
            continue
    raise HTTPException(
        status_code=400,
        detail=f"Path must be under one of allowed roots: {[str(r) for r in roots]}. Got: {str(p)!r}",
    )


class _Missing:
    pass


_MISSING = _Missing()


def _json_type(v: Any) -> str:
    if v is _MISSING:
        return "missing"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    # Fallback for odd types (e.g. Decimal) - treat as string repr
    return "unknown"


def _schema_types(schema: Dict[str, Any]) -> List[str]:
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    # Infer from keywords
    if isinstance(schema.get("properties"), dict):
        return ["object"]
    if "items" in schema:
        return ["array"]
    return []


def _schema_allows_type(schema: Dict[str, Any], vtype: str, root_schema: Dict[str, Any]) -> bool:
    schema = _deref_schema(schema, root_schema)
    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            return any(_schema_allows_type(s, vtype, root_schema) for s in schema[key] if isinstance(s, dict))
    if "allOf" in schema and isinstance(schema["allOf"], list) and schema["allOf"]:
        # allOf: value must satisfy all; use a weak check (any) to avoid false negatives
        return any(_schema_allows_type(s, vtype, root_schema) for s in schema["allOf"] if isinstance(s, dict))

    types = _schema_types(schema)
    if not types:
        return True  # no type constraint
    if vtype == "integer" and "number" in types:
        return True
    return vtype in types


def _pick_variant(schema: Dict[str, Any], value: Any, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = _deref_schema(schema, root_schema)

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            vtype = _json_type(value)
            # Try match by GT type first; fall back to first
            for s in variants:
                if isinstance(s, dict) and _schema_allows_type(s, vtype, root_schema):
                    return _deref_schema(s, root_schema)
            # If gt is missing/null but pred has value, try pred type
            if value in (_MISSING, None):
                return _deref_schema(variants[0], root_schema) if isinstance(variants[0], dict) else schema
            return _deref_schema(variants[0], root_schema) if isinstance(variants[0], dict) else schema

    # allOf: choose first (we don't attempt true merging here)
    if "allOf" in schema and isinstance(schema["allOf"], list) and schema["allOf"]:
        first = schema["allOf"][0]
        return _deref_schema(first, root_schema) if isinstance(first, dict) else schema

    return schema


def _deref_schema(schema: Dict[str, Any], root_schema: Dict[str, Any]) -> Dict[str, Any]:
    if "$ref" in schema and isinstance(schema["$ref"], str):
        resolved = _resolve_ref(schema["$ref"], root_schema)
        if resolved:
            return _deref_schema(resolved, root_schema)
    return schema


def _safe_scalar(v: Any, max_len: int = 200) -> Any:
    # Keep response JSON small and stable.
    if v is _MISSING:
        return "<MISSING>"
    if isinstance(v, (str, int, float, bool)) or v is None:
        if isinstance(v, str) and len(v) > max_len:
            return v[: max_len - 3] + "..."
        return v
    # objects/arrays: truncated JSON
    try:
        s = json.dumps(v, ensure_ascii=False)
    except Exception:
        s = repr(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _get_prop(obj: Any, key: str, *, key_fallbacks: bool = False) -> Any:
    if not isinstance(obj, dict):
        return _MISSING
    if key in obj:
        return obj[key]
    if key_fallbacks and key.endswith("_raw"):
        k2 = key[:-4]
        if k2 in obj:
            return obj[k2]
    return _MISSING


def _get_decimal_sep() -> str:
    """Decimal separator preference for numeric parsing.

    OCRLTY_DECIMAL_SEP:
      - "." (default)    : dot is decimal separator
      - ","              : comma is decimal separator
      - "auto"           : infer from the string

    """
    v = (os.getenv("OCRLTY_DECIMAL_SEP", ".") or ".").strip().lower()
    if v in {".", ","}:
        return v
    return "auto"


def canon_x_string(v: Any, *, mode: str = "strip_collapse") -> Tuple[bool, Optional[str], Optional[str]]:
    """x-string canonicalization.

    - None -> ok, None
    - other -> stringified + NFKC normalized
    - mode: exact | strip | strip_collapse
    - empty string -> None (treat as null-like noise)
    """
    if v is None:
        return True, None, None

    try:
        s = v if isinstance(v, str) else str(v)
    except Exception:
        return False, None, "to_string_failed"

    s = unicodedata.normalize("NFKC", s)

    if mode == "exact":
        out = s
    else:
        out = s.strip()
        if mode == "strip_collapse":
            out = re.sub(r"\s+", " ", out)

    if out == "":
        return True, None, None

    return True, out, None


def canon_x_count(v: Any) -> Tuple[bool, Optional[int], Optional[str]]:
    """x-count canonicalization (quantity / count).

    Accepts:
      - int
      - float (only if int-like)
      - string (extracts first integer anywhere: "1 X" -> 1, "x2" -> 2)
      - None -> ok, None
    """
    if v is None:
        return True, None, None

    if isinstance(v, bool):
        return False, None, "bool_not_allowed"

    if isinstance(v, int) and not isinstance(v, bool):
        return True, int(v), None

    if isinstance(v, float):
        if abs(v - round(v)) <= 1e-9:
            return True, int(round(v)), None
        return False, None, "float_not_intlike"

    if not isinstance(v, str):
        try:
            v = str(v)
        except Exception:
            return False, None, "to_string_failed"

    s = v.strip()
    if s == "":
        return True, None, None

    m = re.search(r"[+-]?\d+", s)
    if not m:
        return False, None, "no_int_found"

    try:
        return True, int(m.group(0)), None
    except Exception:
        return False, None, "int_parse_failed"


def canon_x_money(v: Any) -> Tuple[bool, Optional[float], Optional[str]]:
    """x-money canonicalization.

    Goal: turn things like "261,333", "Rp 73.450", "1 234,56" into a number.

    - None -> ok, None
    - int/float -> float
    - string -> parse with OCRLTY_DECIMAL_SEP (auto/. /,)
    """
    if v is None:
        return True, None, None

    if isinstance(v, bool):
        return False, None, "bool_not_allowed"

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True, float(v), None

    if not isinstance(v, str):
        try:
            v = str(v)
        except Exception:
            return False, None, "to_string_failed"

    s = v.strip()
    if s == "":
        return True, None, None

    # Keep digits, separators, sign; drop currency/letters/spaces.
    s = re.sub(r"[^0-9,\.\-+]", "", s)
    if not re.search(r"\d", s):
        return False, None, "no_digits"

    dec_pref = _get_decimal_sep()

    def _collapse_dots(x: str) -> str:
        # Collapse multiple dots to a single decimal dot (keep last as decimal).
        if x.count(".") <= 1:
            return x
        parts = x.split(".")
        return "".join(parts[:-1]) + "." + parts[-1]

    def parse_with_decimal_sep(raw: str, dec: str) -> Optional[float]:
        other = "," if dec == "." else "."
        x = raw.replace(other, "")
        if dec == ",":
            x = x.replace(",", ".")
        x = _collapse_dots(x)
        try:
            return float(x)
        except Exception:
            return None

    def parse_auto(raw: str) -> Optional[float]:
        x = raw
        has_comma = "," in x
        has_dot = "." in x

        if has_comma and has_dot:
            # right-most separator is decimal, the other is thousands
            dec = "." if x.rfind(".") > x.rfind(",") else ","
            return parse_with_decimal_sep(x, dec)

        if has_comma:
            left, right = x.rsplit(",", 1)
            # if 1-2 digits after comma -> likely decimals
            if re.fullmatch(r"\d{1,2}", right or ""):
                return parse_with_decimal_sep(x, ",")
            # if exactly 3 digits -> likely thousands
            if re.fullmatch(r"\d{3}", right or ""):
                try:
                    return float((left + right).replace(",", ""))
                except Exception:
                    return None
            # fallback: treat commas as thousands
            try:
                return float(x.replace(",", ""))
            except Exception:
                return None

        if has_dot:
            # multiple dots: likely thousands separators
            if x.count(".") > 1:
                try:
                    return float(x.replace(".", ""))
                except Exception:
                    return None
            left, right = x.rsplit(".", 1)
            # if exactly 3 digits -> thousands
            if re.fullmatch(r"\d{3}", right or ""):
                try:
                    return float(left + right)
                except Exception:
                    return None
            return parse_with_decimal_sep(x, ".")

        try:
            return float(x)
        except Exception:
            return None

    out: Optional[float]
    if dec_pref in {".", ","}:
        out = parse_with_decimal_sep(s, dec_pref)
    else:
        out = parse_auto(s)

    if out is None:
        return False, None, "parse_fail"
    return True, out, None


def _canon_value(v: Any, *, fmt: Optional[str], str_mode: str) -> Tuple[bool, Any, Optional[str]]:
    """Canonicalize a scalar value given a schema format hint."""
    if v is _MISSING:
        return False, None, "missing"

    f = (fmt or "").strip().lower()
    if f in {"x-string", "x-str", "string"}:
        return canon_x_string(v, mode=str_mode)
    if f in {"x-count", "x-int", "x-integer", "count"}:
        return canon_x_count(v)
    if f in {"x-money", "x-number", "money", "number"}:
        return canon_x_money(v)

    # No explicit format -> conservative fallback.
    if isinstance(v, str) or v is None:
        return canon_x_string(v, mode=str_mode)

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True, float(v), None

    # last resort: stringify
    return canon_x_string(v, mode=str_mode)


def _compare_canon(
    pred_raw: Any,
    gt_raw: Any,
    *,
    fmt: Optional[str],
    str_mode: str,
) -> Tuple[bool, str, Any, Any]:
    """Compare leaf scalar values with canonicalization.

    Returns:
      (matched, reason, pred_value_canon, gt_value_canon)
    """
    if gt_raw is _MISSING:
        return False, "gt_missing", None, None
    if pred_raw is _MISSING:
        return False, "pred_missing", None, None

    ok_gt, gt_v, gt_err = _canon_value(gt_raw, fmt=fmt, str_mode=str_mode)
    ok_pred, pred_v, pred_err = _canon_value(pred_raw, fmt=fmt, str_mode=str_mode)

    if not ok_gt and not ok_pred:
        return False, "canon_errors", pred_v, gt_v
    if not ok_gt:
        return False, "gt_canon_error", pred_v, gt_v
    if not ok_pred:
        return False, "pred_canon_error", pred_v, gt_v

    if gt_v is None and pred_v is None:
        return True, "ok", pred_v, gt_v
    if (gt_v is None) != (pred_v is None):
        return False, "value_mismatch", pred_v, gt_v

    f = (fmt or "").strip().lower()
    if f in {"x-money", "x-number", "money", "number"} and isinstance(gt_v, (int, float)) and isinstance(pred_v, (int, float)):
        if math.isclose(float(gt_v), float(pred_v), rel_tol=0.0, abs_tol=1e-6):
            return True, "ok", pred_v, gt_v
        return False, "value_mismatch", pred_v, gt_v

    if gt_v == pred_v:
        return True, "ok", pred_v, gt_v
    return False, "value_mismatch", pred_v, gt_v


def _walk_schema(
    schema: Dict[str, Any],
    pred: Any,
    gt: Any,
    *,
    root_schema: Dict[str, Any],
    path: Tuple[Union[str, int, None], ...],
    metrics: Dict[str, Dict[str, int]],
    mismatches: List[Dict[str, Any]],
    str_mode: str,
    key_fallbacks: bool,
    mismatch_limit: int,
) -> None:
    schema_eff = _pick_variant(schema, gt, root_schema)
    schema_eff = _deref_schema(schema_eff, root_schema)

    # Determine node type
    types = _schema_types(schema_eff)
    node_type = types[0] if types else _json_type(gt if gt is not _MISSING else pred)
    if node_type == "integer" and isinstance(gt, float):
        node_type = "number"

    if node_type == "object":
        props = schema_eff.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        pred_obj = pred if isinstance(pred, dict) else {}
        gt_obj = gt if isinstance(gt, dict) else {}

        for k, subschema in props.items():
            if not isinstance(subschema, dict):
                continue
            pred_v = _get_prop(pred_obj, k, key_fallbacks=key_fallbacks)
            gt_v = _get_prop(gt_obj, k, key_fallbacks=key_fallbacks)
            _walk_schema(
                subschema,
                pred_v,
                gt_v,
                root_schema=root_schema,
                path=path + (k,),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=str_mode,
                key_fallbacks=key_fallbacks,
                mismatch_limit=mismatch_limit,
            )
        return

    if node_type == "array":
        items_schema = schema_eff.get("items", {})
        if not isinstance(items_schema, dict):
            items_schema = {}
        pred_list = pred if isinstance(pred, list) else []
        gt_list = gt if isinstance(gt, list) else []
        n = max(len(pred_list), len(gt_list))
        for i in range(n):
            pred_v = pred_list[i] if i < len(pred_list) else _MISSING
            gt_v = gt_list[i] if i < len(gt_list) else _MISSING
            _walk_schema(
                items_schema,
                pred_v,
                gt_v,
                root_schema=root_schema,
                path=path + (None,),  # wildcard index
                metrics=metrics,
                mismatches=mismatches,
                str_mode=str_mode,
                key_fallbacks=key_fallbacks,
                mismatch_limit=mismatch_limit,
            )
        return

    # Scalar leaf
    def _fmt_path(pth: Tuple[Union[str, int, None], ...]) -> str:
        out = ""
        for seg in pth:
            if seg is None:
                out += "[*]"
            elif isinstance(seg, int):
                out += f"[{seg}]"
            else:
                out += ("." if out else "") + str(seg)
        return out or "<root>"

    path_str = _fmt_path(path)
    m = metrics.setdefault(path_str, defaultdict(int))
    m["total"] += 1

    if gt is _MISSING:
        m["gt_missing"] += 1
        return

    m["comparable"] += 1

    if pred is _MISSING:
        m["pred_missing"] += 1
        m["mismatched"] += 1
        if len(mismatches) < mismatch_limit:
            mismatches.append(
                {
                    "path": path_str,
                    "gt_raw": _safe_scalar(gt),
                    "pred_raw": _safe_scalar(pred),
                    "gt_value": None,
                    "pred_value": None,
                    "reason": "pred_missing",
                }
            )
        return

    fmt = schema_eff.get("format") if isinstance(schema_eff.get("format"), str) else None

    matched, reason, pred_v, gt_v = _compare_canon(pred, gt, fmt=fmt, str_mode=str_mode)

    if matched:
        m["matched"] += 1
        return

    m["mismatched"] += 1
    if reason == "gt_canon_error":
        m["gt_canon_error"] += 1
    elif reason == "pred_canon_error":
        m["pred_canon_error"] += 1
    elif reason == "canon_errors":
        m["canon_errors"] += 1
    else:
        m["value_mismatch"] += 1

    if len(mismatches) < mismatch_limit:
        mismatches.append(
            {
                "path": path_str,
                "gt_raw": _safe_scalar(gt),
                "pred_raw": _safe_scalar(pred),
                "gt_value": _safe_scalar(gt_v),
                "pred_value": _safe_scalar(pred_v),
                "reason": reason,
                "format": fmt,
            }
        )


def _schema_leaf_paths(schema: Dict[str, Any], root_schema: Dict[str, Any]=None, *, path: Tuple[Union[str, None], ...] = ()) -> List[str]:
    """Return leaf (scalar) paths derived from schema, mostly for reporting."""
    if root_schema is None:
        root_schema = schema
    schema_eff = _deref_schema(_pick_variant(schema, _MISSING, root_schema), root_schema)
    types = _schema_types(schema_eff)
    node_type = types[0] if types else None

    if node_type == "object" or isinstance(schema_eff.get("properties"), dict):
        out: List[str] = []
        props = schema_eff.get("properties", {})
        if isinstance(props, dict):
            for k, s in props.items():
                if isinstance(s, dict):
                    out.extend(_schema_leaf_paths(s, root_schema, path=path + (k,)))
        return out

    if node_type == "array" or "items" in schema_eff:
        items_schema = schema_eff.get("items", {})
        if isinstance(items_schema, dict):
            return _schema_leaf_paths(items_schema, root_schema, path=path + (None,))
        return []

    # Scalar
    out = ""
    for seg in path:
        if seg is None:
            out += "[*]"
        else:
            out += ("." if out else "") + str(seg)
    return [out or "<root>"]


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

@app.get(
    "/v1/debug/artifacts", 
    response_model=ArtifactListResponse,
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],)
async def list_artifacts(
    date: Optional[str] = Query(default=None, description="Filter by day (YYYY-MM-DD). If omitted, returns most recent items across days."),
    limit: int = Query(default=50, ge=1, le=500),
) -> ArtifactListResponse:
    items: List[ArtifactListItem] = []

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")

        if _s3_enabled():
            prefix = str(_extract_artifact_ref("dummy", day=date)).replace("dummy.json", "")
            objs = _s3_list_objects(prefix, limit=limit * 3)
            objs.sort(key=lambda x: x.get("LastModified") or datetime.fromtimestamp(0, tz=timezone.utc), reverse=True)
            for it in objs[:limit]:
                key = it.get("Key") or ""
                rid = Path(key).stem
                meta = ArtifactListItem(request_id=rid, date=date)
                try:
                    data = _s3_get_json(key)
                    meta.task_id = data.get("task_id")
                    meta.schema_ok = data.get("schema_ok")
                except Exception:
                    pass
                items.append(meta)
            return ArtifactListResponse(items=items)

        day_dir = (ARTIFACTS_DIR / "extracts") / date
        if not day_dir.exists():
            return ArtifactListResponse(items=[])

        files = sorted(day_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:limit]:
            rid = p.stem
            meta = ArtifactListItem(request_id=rid, date=date)
            try:
                data = _read_artifact_json(p)
                meta.task_id = data.get("task_id")
                meta.schema_ok = data.get("schema_ok")
            except Exception:
                pass
            items.append(meta)

        return ArtifactListResponse(items=items)

    # no date: walk recent day folders and collect
    if _s3_enabled():
        for d in _iter_artifact_days_desc():
            prefix = str(_extract_artifact_ref("dummy", day=d)).replace("dummy.json", "")
            objs = _s3_list_objects(prefix, limit=limit * 3)
            objs.sort(key=lambda x: x.get("LastModified") or datetime.fromtimestamp(0, tz=timezone.utc), reverse=True)
            for it in objs:
                key = it.get("Key") or ""
                rid = Path(key).stem
                meta = ArtifactListItem(request_id=rid, date=d)
                try:
                    data = _s3_get_json(key)
                    meta.task_id = data.get("task_id")
                    meta.schema_ok = data.get("schema_ok")
                except Exception:
                    pass
                items.append(meta)
                if len(items) >= limit:
                    return ArtifactListResponse(items=items)
        return ArtifactListResponse(items=items)

    for d in _iter_artifact_days_desc():
        day_dir = (ARTIFACTS_DIR / "extracts") / d
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

    # no date: walk recent day folders and collect
    for d in _iter_artifact_days_desc():
        day_dir = (ARTIFACTS_DIR / "extracts") / d
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



@app.get(
    "/v1/debug/artifacts/backup.tar.gz",
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],
    )
async def get_artifacts_backup(
    background_tasks: BackgroundTasks,
):
    """
    Create a tar.gz backup of artifacts and return it as a download.

    NOTE: This endpoint is debug-only and requires DEBUG_MODE=1 and debug:read_raw scope.
    In S3 mode, this downloads objects into a temporary tar.gz on-demand (may be slow for large buckets).
    """
    fd, tmp_path = tempfile.mkstemp(prefix="artifacts_backup_", suffix=".tar.gz", dir="/tmp")
    os.close(fd)

    def _build_tar_local() -> None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(str(ARTIFACTS_DIR), arcname=str(ARTIFACTS_DIR.name), recursive=True)

    def _build_tar_s3() -> None:
        max_objects = int(os.getenv("DEBUG_BACKUP_MAX_OBJECTS", "2000"))
        prefix = _s3_key("")  # root
        objs = _s3_list_objects(prefix, limit=max_objects + 1)
        if len(objs) > max_objects:
            raise RuntimeError(
                f"Too many objects for backup (>{max_objects}). Increase DEBUG_BACKUP_MAX_OBJECTS if needed."
            )
        with tarfile.open(tmp_path, "w:gz") as tar:
            for it in objs:
                key = it.get("Key") or ""
                if not key or key.endswith("/"):
                    continue
                rel = key
                if _S3_PREFIX and rel.startswith(_S3_PREFIX + "/"):
                    rel = rel[len(_S3_PREFIX) + 1 :]
                data = _s3_get_text(key).encode("utf-8", errors="replace")
                info = tarfile.TarInfo(name=rel)
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, BytesIO(data))

    try:
        if _s3_enabled():
            await asyncio.to_thread(_build_tar_s3)
        else:
            await asyncio.to_thread(_build_tar_local)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise

    background_tasks.add_task(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))

    return FileResponse(
        tmp_path,
        media_type="application/gzip",
        filename="artifacts_backup.tar.gz",
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/v1/debug/artifacts/{request_id}",
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],
    )
async def read_artifact(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
) -> Dict[str, Any]:
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


@app.get(
    "/v1/debug/artifacts/{request_id}/raw",
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],
    )
async def read_artifact_raw_text(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
) -> Dict[str, Any]:
    p = _find_artifact_path(request_id, date=date)
    if p is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    data = _read_artifact_json(p)
    return {
        "request_id": request_id,
        "date": _artifact_date_from_path(p),
        "raw_model_text": data.get("raw_model_text"),
    }



# --- p.3b: debug evaluation (batch artifacts vs GT) ---

class EvalBatchVsGTRequest(BaseModel):
    run_id: str = Field(..., description="Batch run_id to load from artifacts/batches/<day>/<run_id>.json")
    batch_date: Optional[str] = Field(default=None, description="Optional day (YYYY-MM-DD) to avoid searching over days.")
    max_days: int = Field(default=30, ge=1, le=365, description="How many recent days to search for batch artifact if batch_date is not provided.")
    gt_path: str = Field(..., description="Path to GT JSON file (must be under DATA_ROOT or ARTIFACTS_DIR).")
    gt_image_key: str = Field(default="file", description="Key inside each GT record that contains the image path/name (basename is used). Fallback: image_file.")
    gt_record_key: Optional[str] = Field(default=None, description="Optional key inside each GT record to use as the actual GT object (e.g. 'gt' or 'fields').")
    limit: Optional[int] = Field(default=None, ge=1, description="Optional max number of batch items to evaluate.")

    # Output size controls (keep endpoint ergonomic).
    include_worst: bool = Field(default=False, description="Include worst_samples only in the saved eval artifact instead of all samples.")
    samples_limit: int = Field(default=10, ge=0, le=200, description="How many worst samples to include (0 disables).")
    mismatches_per_sample: int = Field(default=20, ge=0, le=200, description="Max mismatches to include per sample.")

    # Canonicalization controls
    str_mode: str = Field(default="strip_collapse", description="x-string mode: exact | strip | strip_collapse")


class EvalFieldMetric(BaseModel):
    path: str
    total: int
    comparable: int
    matched: int
    mismatched: int
    accuracy: Optional[float] = None

    pred_missing: int = 0
    gt_missing: int = 0

    gt_canon_error: int = 0
    pred_canon_error: int = 0
    canon_errors: int = 0
    value_mismatch: int = 0


class EvalSampleSummary(BaseModel):
    image: str
    request_id: str
    ok: bool
    schema_ok: bool
    artifact_rel: Optional[str] = None
    gt_found: bool = False
    pred_found: bool = False
    mismatches: int = 0
    mismatches_preview: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class EvalBatchVsGTResult(BaseModel):
    eval_id: str
    run_id: str
    batch_date: Optional[str] = None
    task_id: Optional[str] = None
    model: Optional[str] = None
    inference_backend: Optional[str] = None

    batch_summary: Dict[str, Any]
    gt_summary: Dict[str, Any]
    eval_summary: Dict[str, Any]

    fields: List[EvalFieldMetric]
    worst_samples: Optional[List[EvalSampleSummary]] = None


class EvalBatchVsGTResponse(BaseModel):
    eval_id: str #eval_id,
    created_at: str #datetime.now(timezone.utc).isoformat(),
    run_id: str #run_id,
    gt_path: str #req.gt_path,
    batch_artifact_path: str
    eval_artifact_path: str #eval_artifact_path,


@app.post(
    "/v1/debug/eval/batch_vs_gt",
    response_model=EvalBatchVsGTResponse,
    dependencies=[Depends(require_scopes(["debug:run"]))]
)
async def eval_batch_vs_gt(
    req: EvalBatchVsGTRequest,
) -> EvalBatchVsGTResponse:
    # Validate
    if req.str_mode not in {"exact", "strip", "strip_collapse"}:
        raise HTTPException(status_code=400, detail="str_mode must be one of: exact | strip | strip_collapse")

    # Locate batch artifact
    batch_path = _find_batch_artifact_path(req.run_id, date=req.batch_date, max_days=req.max_days)
    if batch_path is None:
        raise HTTPException(status_code=404, detail="Batch artifact not found (check run_id and batch_date/max_days).")

    batch_obj = _read_json_file(batch_path)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid batch artifact JSON (expected object).")

    batch_date = _artifact_date_from_path(batch_path) or (req.batch_date or "")
    task_id = str(batch_obj.get("task_id") or "")
    if not task_id:
        raise HTTPException(status_code=500, detail="Batch artifact missing task_id.")
    task = _get_task(task_id)

    # Load schema JSON for the task
    schema_json = task.get("schema_value")
    if not isinstance(schema_json, dict):
        raise HTTPException(status_code=500, detail=f"Task {task_id!r} has no schema_value (dict).")

    leaf_paths = _schema_leaf_paths(schema_json)

    # Load GT
    gt_p = _validate_under_any_root(Path(req.gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])
    gt_obj = _read_json_file(gt_p)

    # GT may be {id: rec} or [rec, ...]
    if isinstance(gt_obj, dict):
        gt_records = list(gt_obj.values())
    elif isinstance(gt_obj, list):
        gt_records = gt_obj
    else:
        raise HTTPException(status_code=400, detail="GT JSON must be an object or an array of objects.")

    gt_index: Dict[str, Dict[str, Any]] = {}
    for rec in gt_records:
        if not isinstance(rec, dict):
            continue
        img_val = rec.get(req.gt_image_key)
        if img_val is None and req.gt_image_key != "image_file":
            img_val = rec.get("image_file")
        if img_val is None:
            continue
        base = Path(str(img_val)).name
        if base:
            gt_index[base] = rec

    # Evaluate
    items = batch_obj.get("items") if isinstance(batch_obj.get("items"), list) else []
    if req.limit is not None:
        items = items[: int(req.limit)]

    metrics: Dict[str, Dict[str, int]] = {}
    sample_summaries: List[Dict[str, Any]] = []

    gt_found = 0
    gt_missing = 0
    pred_found = 0
    pred_missing = 0

    mismatch_limit_total = 5000  # hard cap per sample to keep eval artifacts bounded

    for it in items:
        if not isinstance(it, dict):
            continue

        image = it.get("file") or it.get("image") or it.get("image_file")
        image = str(image) if image is not None else ""
        image_base = Path(image).name if image else ""

        request_id = str(it.get("request_id") or "")
        artifact_rel = it.get("artifact_rel")
        ok = bool(it.get("ok"))
        schema_ok = bool(it.get("schema_ok"))

        gt_rec = gt_index.get(image_base) if image_base else None
        gt_ok = gt_rec is not None
        if gt_ok:
            gt_found += 1
        else:
            gt_missing += 1

        # Resolve pred artifact
        pred_obj: Any = _MISSING
        artifact_abs: Optional[Path] = None
        if isinstance(artifact_rel, str) and artifact_rel:
            rel = Path(artifact_rel)
            if rel.is_absolute():
                artifact_abs = _validate_under_any_root(rel, roots=[ARTIFACTS_DIR, DATA_ROOT])
            else:
                # Old batch format might omit "extracts/" prefix.
                if len(rel.parts) >= 1 and _DATE_RE.match(rel.parts[0]) and rel.parts[0].count("-") == 2:
                    rel = Path("extracts") / rel
                artifact_abs = (ARTIFACTS_DIR / rel).resolve()
        if artifact_abs is not None and artifact_abs.exists():
            art = _read_artifact_json(artifact_abs)
            # Common locations for model output
            pred_obj = (
                art.get("parsed")
                or art.get("result")
                or art.get("data")
                or art.get("output")
                or art.get("extracted")
                or _MISSING
            )
            if pred_obj is not _MISSING:
                pred_found += 1
            else:
                pred_missing += 1
        else:
            pred_missing += 1

        # Choose GT payload object
        gt_payload: Any = gt_rec if gt_rec is not None else _MISSING
        if gt_payload is not _MISSING and req.gt_record_key:
            if isinstance(gt_payload, dict) and req.gt_record_key in gt_payload:
                gt_payload = gt_payload.get(req.gt_record_key)
            else:
                # If requested key is missing, treat as missing GT for scoring.
                gt_payload = _MISSING

        mismatches: List[Dict[str, Any]] = []

        if gt_payload is not _MISSING and pred_obj is not _MISSING:
            _walk_schema(
                schema_json,
                pred_obj,
                gt_payload,
                root_schema=schema_json,
                path=(),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=req.str_mode,
                key_fallbacks=False,
                mismatch_limit=mismatch_limit_total,
            )
        elif gt_payload is not _MISSING and pred_obj is _MISSING:
            # Count missing pred for every leaf path in schema
            for pth in leaf_paths:
                m = metrics.setdefault(pth, defaultdict(int))
                m["total"] += 1
                m["comparable"] += 1
                m["pred_missing"] += 1
                m["mismatched"] += 1

        # Build sample summary
        mismatches_preview = mismatches[: int(req.mismatches_per_sample)] if req.mismatches_per_sample > 0 else []

        sample_summaries.append(
            {
                "image": image_base or image,
                "request_id": request_id,
                "ok": ok,
                "schema_ok": schema_ok,
                "artifact_rel": str(artifact_rel) if artifact_rel is not None else None,
                "gt_found": gt_ok,
                "pred_found": pred_obj is not _MISSING,
                "mismatches": len(mismatches),
                "mismatches_preview": mismatches_preview,
                "notes": [],
            }
        )

    # Build per-field metrics (ordered by schema leaf paths if possible)
    order = {p: i for i, p in enumerate(leaf_paths)}
    field_rows: List[EvalFieldMetric] = []
    for path_str, m in metrics.items():
        total = int(m.get("total", 0))
        comparable = int(m.get("comparable", 0))
        matched = int(m.get("matched", 0))
        mismatched = int(m.get("mismatched", 0))
        acc = (matched / comparable) if comparable else None

        field_rows.append(
            EvalFieldMetric(
                path=path_str,
                total=total,
                comparable=comparable,
                matched=matched,
                mismatched=mismatched,
                accuracy=acc,
                pred_missing=int(m.get("pred_missing", 0)),
                gt_missing=int(m.get("gt_missing", 0)),
                gt_canon_error=int(m.get("gt_canon_error", 0)),
                pred_canon_error=int(m.get("pred_canon_error", 0)),
                canon_errors=int(m.get("canon_errors", 0)),
                value_mismatch=int(m.get("value_mismatch", 0)),
            )
        )

    field_rows.sort(key=lambda r: (order.get(r.path, 10**9), r.path))

    comparable_total = sum(r.comparable for r in field_rows)
    matched_total = sum(r.matched for r in field_rows)
    overall_accuracy = (matched_total / comparable_total) if comparable_total else None

    # Pick worst samples (by mismatches)
    worst_samples: Optional[List[Dict[str, Any]]] = None
    if req.include_worst and req.samples_limit > 0:
        worst = [s for s in sample_summaries if int(s.get("mismatches") or 0) > 0]
        worst.sort(key=lambda s: int(s.get("mismatches") or 0), reverse=True)
        worst_samples = worst[: int(req.samples_limit)]
        sample_summaries = []

    eval_id = f"{req.run_id}__eval_{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()

    payload: Dict[str, Any] = {
        "eval_id": eval_id,
        "created_at": created_at,
        "run_id": req.run_id,
        "gt_path": str(gt_p),
        "batch_artifact_path": str(batch_path),
        "batch_date": batch_date,
        "task_id": task_id,
        "model": batch_obj.get("model"),
        "inference_backend": batch_obj.get("inference_backend"),
        "batch_summary": batch_obj.get("summary", {}),
        "gt_summary": {
            "gt_path": str(gt_p),
            "gt_records": len(gt_index),
            "gt_image_key": req.gt_image_key,
            "gt_found_for_batch_items": gt_found,
            "gt_missing_for_batch_items": gt_missing,
        },
        "eval_summary": {
            "items_considered": len(items),
            "pred_found": pred_found,
            "pred_missing": pred_missing,
            "leaf_paths_in_schema": len(leaf_paths),
            "comparable_total": comparable_total,
            "matched_total": matched_total,
            "overall_accuracy": overall_accuracy,
            "str_mode": req.str_mode,
            "decimal_sep": _get_decimal_sep(),
        },
        "fields": [r.model_dump() for r in field_rows],
        "worst_samples": worst_samples,
        "all_samples": sample_summaries,
    }

    eval_artifact_path = _save_eval_artifact(payload)

    return EvalBatchVsGTResponse(
        eval_id=eval_id,
        created_at=created_at,
        run_id=req.run_id,
        gt_path=str(gt_p),
        batch_artifact_path=str(batch_path),
        eval_artifact_path=eval_artifact_path,
    )


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

    # Record system-level errors/retries in artifacts (and optionally return them).
    error_history: List[Dict[str, Any]] = []
    image_resize: Optional[Dict[str, Any]] = None

    messages_cur = messages
    mime_cur = req.mime_type

    t_inf0 = time.perf_counter()
    t_inf1 = t_inf0

    retried = False
    attempt = 0

    prompt_sha256 = hashlib.sha256(task["prompt_value"].encode("utf-8")).hexdigest()
    schema_obj = task["schema_value"]  
    schema_canon = json.dumps(schema_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    schema_sha256 = hashlib.sha256(schema_canon.encode("utf-8")).hexdigest()

    artifact_base = {
    "request_id": request_id,
    "task_id": req.task_id,
    "model": MODEL_ID,
    "inference_backend": backend,
    "input": _sanitize_input_for_artifact(req, image_resize=image_resize),
    "auth": {"key_id": principal.key_id, "role": principal.role},
    "image_resize": image_resize,
    "prompt_sha256": prompt_sha256,
    "schema_sha256": schema_sha256,
    }

    while True:
        try:
            t_inf0 = time.perf_counter()
            if backend == "mock":
                raw = _mock_chat_completions(request_id, principal, req.debug, schema_dict=schema_dict)
            else:
                raw = await vllm.chat_completions(
                    model=MODEL_ID,
                    messages=messages_cur,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    request_id=request_id,
                )
            t_inf1 = time.perf_counter()
            break

        except Exception as e:
            err = str(e)
            error_history.append(_mk_err("inference_request" if attempt == 0 else "retry_request", err))

            # One-shot smart retry: if server says the decoder prompt is too long, downscale the image
            # by a factor derived from server-provided token counts.
            if attempt == 0 and backend != "mock":
                ctx = _parse_decoder_prompt_too_long(err)
                if ctx and req.image_base64 and (not req.image_url):
                    t_resp, t_max = ctx
                    t_prompt_est = _estimate_prompt_tokens(prompt_text, schema_dict)
                    scale = _compute_retry_scale(t_resp=t_resp, t_max=t_max, t_prompt_est=t_prompt_est)
                    if scale is None:
                        error_history.append(
                            _mk_err(
                                "retry_skipped",
                                "Context-length error but estimated prompt is already too large; image downscale cannot help.",
                                t_resp=t_resp,
                                t_max=t_max,
                                t_prompt_est=t_prompt_est,
                            )
                        )
                    else:
                        try:
                            b64_new, resize_info = _resize_base64_image_by_scale(req.image_base64, req.mime_type, scale)
                            if resize_info.get("applied"):
                                image_resize = {
                                    **resize_info,
                                    "t_resp": int(t_resp),
                                    "t_max": int(t_max),
                                    "t_prompt_est": int(t_prompt_est),
                                }

                                mime_cur = image_resize.get("sent_mime_type") or req.mime_type
                                messages_cur = [
                                    {
                                        "role": "user",
                                        "content": [
                                            _image_content(req.image_url, b64_new, mime_cur),
                                            {"type": "text", "text": prompt_text},
                                        ],
                                    }
                                ]

                                retried = True
                                attempt = 1
                                continue

                            error_history.append(
                                _mk_err(
                                    "retry_skipped",
                                    "Computed resize scale resulted in no size change; skipping retry.",
                                    scale=float(scale),
                                )
                            )
                        except Exception as e2:
                            error_history.append(_mk_err("retry_failed", str(e2)))
            

            # Inference failed (and retry either didn't happen or also failed).
            artifact_path = _save_artifact(
                request_id,
                {
                    **artifact_base,
                    "error_history": error_history,
                    "error_stage": "inference_request",
                    "vllm_request_meta": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "has_response_format": response_format is not None,
                        "retried": retried,
                    },
                    "timings_ms": {
                        "total_ms": (time.perf_counter() - t0) * 1000.0,
                    },
                },
            )
            detail = f"Inference request failed ({backend}): {err}"
            if _raw_allowed(principal):
                detail += f" | artifact_rel={_to_artifact_rel(artifact_path)}"
            raise HTTPException(status_code=502, detail=detail)

    try:
        raw_text = raw["choices"][0]["message"]["content"]
    except Exception as e:
        error_history.append(_mk_err("unexpected_response_format", str(e)))

        artifact_path = _save_artifact(
            request_id,
            {
                **artifact_base,
                "error_stage": "unexpected_response_format",
                "error_history": error_history,
                "raw_response": raw,
                "timings_ms": {
                    "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
            },
        )
        detail = f"Unexpected {backend} response format."
        if _raw_allowed(principal):
            detail += f" | artifact_rel={_to_artifact_rel(artifact_path)}"
        raise HTTPException(status_code=502, detail=detail)

    # Parse JSON produced by the model.
    try:
        parsed = json.loads(raw_text)
    except Exception as e:
        error_history.append(_mk_err("parse_json", str(e)))

        # Сохраняем артефакт, иначе реально “некуда посмотреть”
        artifact_path = _save_artifact(
            request_id,
            {
                **artifact_base,
                "error_stage": "parse_json",
                "error_history": error_history,
                "raw_model_text": raw_text,
                "raw_response": raw,
                "vllm_request_meta": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "has_response_format": response_format is not None,
                },
                "timings_ms": {
                    "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
            },
        )
        detail = f"Model output is not valid JSON ({backend}): {e}"
        if _raw_allowed(principal):
            detail += f" | artifact_rel={_to_artifact_rel(artifact_path)}"
        raise HTTPException(status_code=502, detail=detail)


    if schema_dict is None:
        schema_ok = True
        errors: list[str] = []
    else:
        schema_json_validator = _make_validator_or_raise(schema_dict, is_custom_schema=is_custom_schema)
        errors = [e.message for e in schema_json_validator.iter_errors(parsed)]
        schema_ok = len(errors) == 0

    artifact_key = _save_artifact(
        request_id,
        {
            **artifact_base,
            "error_history": error_history,
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
        image_resize=image_resize,
        error_history=error_history,
        artifact_rel=_to_artifact_rel(artifact_key),
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
    task = _get_task(req.task_id)

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

                resp = await extract(sub_req, principal)
                artifact_rel = resp.artifact_rel

                return {
                    "image": rel_path,
                    "request_id": resp.request_id,
                    "artifact_rel": artifact_rel,
                    "ok": True,
                    "schema_ok": resp.schema_ok,
                    "timings_ms": resp.timings_ms,
                    "image_resize": resp.image_resize,
                    "error_history": resp.error_history,
                }

            except HTTPException as he:
                art_p = _find_artifact_path(request_id, max_days=30)
                artifact_rel = _to_artifact_rel(art_p) if art_p is not None else None
                err_hist = _try_load_error_history_for_request(request_id) or [
                    _mk_err("http_error", f"HTTP {he.status_code}: {he.detail}")
                ]
                return {
                    "image": rel_path,
                    "request_id": request_id,
                    "artifact_rel": artifact_rel,
                    "ok": False,
                    "schema_ok": False,
                    "error_history": err_hist,
                }
            except Exception as e:
                art_p = _find_artifact_path(request_id, max_days=30)
                artifact_rel = _to_artifact_rel(art_p) if art_p is not None else None
                err_hist = _try_load_error_history_for_request(request_id) or [
                    _mk_err("exception", f"{e.__class__.__name__}: {e}")
                ]
                return {
                    "image": rel_path,
                    "request_id": request_id,
                    "artifact_rel": artifact_rel,
                    "ok": False,
                    "schema_ok": False,
                    "error_history": err_hist,
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
    
    prompt_sha256 = hashlib.sha256(task["prompt_value"].encode("utf-8")).hexdigest()
    schema_obj = task["schema_value"]  
    schema_canon = json.dumps(schema_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    schema_sha256 = hashlib.sha256(schema_canon.encode("utf-8")).hexdigest()

    batch_artifact_path = _save_batch_artifact(
        run_id,
        {
            "run_id": run_id,
            "task_id": req.task_id,
            "images_dir": str(images_dir),
            "glob": req.glob,
            "exts": req.exts,
            "limit": req.limit,
            "task_snapshot": {
                "prompt_text": task["prompt_value"],
                "schema_json": task["schema_value"],
                "prompt_sha256": prompt_sha256,
                "schema_sha256": schema_sha256,
            },
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
    # --- AUTO-EVAL (optional) ---
    eval_obj: Optional[Dict[str, Any]] = None
    eval_artifact_path: Optional[str] = None
    if req.gt_path:
        # Evaluating against a server-side GT file is considered a DEBUG operation.
        # Eval uses the same permissions as the debug eval endpoint: DEBUG_MODE=1 and scope debug:read_raw.
        _debug_enabled()
        if settings.AUTH_ENABLED and ("debug:read_raw" not in principal.scopes):
            raise HTTPException(status_code=403, detail="Missing required scope: debug:read_raw")

        # Validate GT path early for a clearer error.
        _validate_under_any_root(Path(req.gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])

        # Reuse the same evaluation logic as the eval_batch_vs_gt endpoint.
        eval_req = EvalBatchVsGTRequest(
            run_id=run_id,
            batch_date=Path(batch_artifact_path).parent.name,
            gt_path=req.gt_path,
            gt_image_key=req.gt_image_key or "image",
            include_worst=False,
            str_mode="exact",
        )
        eval_resp = await eval_batch_vs_gt(eval_req)
        eval_obj = eval_resp.model_dump()



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
        eval=eval_obj,
    )
