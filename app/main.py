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
import time
import uuid
import tarfile
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
from collections import defaultdict

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
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / "extracts" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_artifact(request_id: str, payload: Dict[str, Any]) -> str:
    _validate_request_id_for_fs(request_id)
    out_dir = _ensure_artifacts_dir()
    path = out_dir / f"{request_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _to_artifact_rel(p: Union[str, Path]) -> str:
    try:
        pp = Path(p).resolve()
        return str(pp.relative_to(ARTIFACTS_DIR))
    except Exception:
        return str(p)


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


def _ensure_eval_artifacts_dir() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARTIFACTS_DIR / "evals" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_eval_artifact(payload: Dict[str, Any]) -> str:
    eval_id  = payload['eval_id'] or ''
    _validate_request_id_for_fs(eval_id)
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
    base = ARTIFACTS_DIR / "extracts"
    if not base.exists():
        return []
    days = []
    for p in base.iterdir():
        if p.is_dir() and _DATE_RE.match(p.name):
            days.append(p.name)
    days.sort(reverse=True)
    return days


def _find_artifact_path(request_id: str, *, date: Optional[str] = None, max_days: int = 30) -> Optional[Path]:
    _validate_request_id_for_fs(request_id)

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")
        p = (ARTIFACTS_DIR / "extracts") / date / f"{request_id}.json"
        return p if p.exists() else None

    days = _iter_artifact_days_desc()[: max(1, int(max_days))]
    for d in days:
        p = (ARTIFACTS_DIR / "extracts") / d / f"{request_id}.json"
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
# Batch artifacts + evaluation helpers
# ---------------------------

def _iter_batch_days_desc() -> List[str]:
    base = ARTIFACTS_DIR / "batches"
    if not base.exists():
        return []
    days: List[str] = []
    for p in base.iterdir():
        if p.is_dir() and _DATE_RE.match(p.name):
            days.append(p.name)
    days.sort(reverse=True)
    return days


def _find_batch_artifact_path(run_id: str, *, date: Optional[str] = None, max_days: int = 30) -> Optional[Path]:
    _validate_request_id_for_fs(run_id)

    if date is not None:
        if not _DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="batch_date must be in YYYY-MM-DD format.")
        p = (ARTIFACTS_DIR / "batches") / date / f"{run_id}.json"
        return p if p.exists() else None

    days = _iter_batch_days_desc()[: max(1, int(max_days))]
    for d in days:
        p = (ARTIFACTS_DIR / "batches") / d / f"{run_id}.json"
        if p.exists():
            return p
    return None


def _read_json_file(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
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


def _normalize_string(s: str, mode: str) -> str:
    if mode == "exact":
        return s
    out = s.strip()
    if mode == "strip_collapse":
        out = re.sub(r"\s+", " ", out)
    return out


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


def _compare_scalar(pred: Any, gt: Any, *, str_mode: str) -> Tuple[bool, str]:
    # Returns (match, reason_if_mismatch_or_note)
    if gt is _MISSING:
        return False, "gt_missing"
    if pred is _MISSING:
        return False, "pred_missing"

    # Normalize strings if requested
    if isinstance(gt, str) and isinstance(pred, str):
        g2 = _normalize_string(gt, str_mode)
        p2 = _normalize_string(pred, str_mode)
        if g2 == p2:
            return True, "ok"
        return False, "value_mismatch"

    # integer vs number: compare numerically if both numeric
    if isinstance(gt, (int, float)) and isinstance(pred, (int, float)) and not isinstance(gt, bool) and not isinstance(pred, bool):
        if float(gt) == float(pred):
            return True, "ok"
        return False, "value_mismatch"

    if gt == pred:
        return True, "ok"

    # Type mismatch is useful to surface
    if _json_type(gt) != _json_type(pred):
        return False, "type_mismatch"

    return False, "value_mismatch"


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
        # If pred/gt are not dicts (or missing), treat children as missing.
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
    # Path string
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
        # Not comparable; nothing else to do
        return
    m["comparable"] += 1

    ok, reason = _compare_scalar(pred, gt, str_mode=str_mode)
    if ok:
        m["matched"] += 1
        return

    m["mismatched"] += 1
    if pred is _MISSING:
        m["pred_missing"] += 1
    if reason == "type_mismatch":
        m["type_mismatch"] += 1
    if gt is None:
        m["gt_null"] += 1
    if pred is None:
        m["pred_null"] += 1

    if len(mismatches) < mismatch_limit:
        mismatches.append(
            {
                "path": path_str,
                "gt": _safe_scalar(gt),
                "pred": _safe_scalar(pred),
                "reason": reason,
            }
        )


def _schema_leaf_paths(schema: Dict[str, Any], root_schema: Dict[str, Any], *, path: Tuple[Union[str, None], ...] = ()) -> List[str]:
    """Return leaf (scalar) paths derived from schema, mostly for reporting."""
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


def _build_gt_index(gt_obj: Any, *, image_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Build basename -> gt_record mapping.
    Accepts:
    - dict[id -> record]
    - list[record]
    """
    records: List[Dict[str, Any]] = []
    if isinstance(gt_obj, dict):
        for v in gt_obj.values():
            if isinstance(v, dict):
                records.append(v)
    elif isinstance(gt_obj, list):
        for v in gt_obj:
            if isinstance(v, dict):
                records.append(v)

    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        img = r.get(image_key)
        if img is None:
            # Common legacy key
            img = r.get("image_file")
        if img is None:
            continue
        base = os.path.basename(str(img))
        if base and base not in out:
            out[base] = r
    return out


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
        day_dir = (ARTIFACTS_DIR / "extracts") / date
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
    Create a tar.gz backup of the whole ARTIFACTS_DIR and return it as a download.

    NOTE: This endpoint is debug-only and requires DEBUG_MODE=1 and debug:read_raw scope.
    """
    # Ensure artifacts dir exists so we can create an archive even on a fresh instance.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="artifacts_backup_", suffix=".tar.gz", dir="/tmp")
    os.close(fd)

    def _build_tar() -> None:
        with tarfile.open(tmp_path, "w:gz") as tar:
            # Put the top-level folder into the archive for convenient extraction.
            tar.add(str(ARTIFACTS_DIR), arcname=str(ARTIFACTS_DIR.name), recursive=True)

    try:
        await asyncio.to_thread(_build_tar)
    except Exception:
        # Clean up tmp file on error.
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


@app.get("/v1/debug/artifacts/{request_id}")
async def read_artifact(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],
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


@app.get("/v1/debug/artifacts/{request_id}/raw")
async def read_artifact_raw_text(
    request_id: str,
    date: Optional[str] = Query(default=None, description="Optional day (YYYY-MM-DD) to avoid search."),
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],
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
    gt_image_key: str = Field(default="image", description="Key inside each GT record that contains the image path/name (basename is used). Fallback: image_file.")
    gt_record_key: Optional[str] = Field(default=None, description="Optional key inside each GT record to use as the actual GT object (e.g. 'gt' or 'fields').")
    limit: Optional[int] = Field(default=None, ge=1, description="Optional max number of batch items to evaluate.")
    sort_samples: bool = Field(default=False, description="Sort samples by mismatches in response.")
    mismatches_per_sample: int = Field(default=30, ge=0, le=200, description="Max mismatches to include per sample.")
    mismatch_limit_total: int = Field(default=5000, ge=0, le=20000, description="Global cap on stored mismatch entries per sample evaluation (prevents huge responses).")
    str_mode: str = Field(default="exact", description="String compare mode: exact | strip | strip_collapse")
    key_fallbacks: bool = Field(default=False, description="If true, try generic key fallbacks like '<name>_raw' -> '<name>' when reading GT/pred objects.")


class EvalFieldMetric(BaseModel):
    path: str
    total: int
    comparable: int
    matched: int
    mismatched: int
    accuracy: Optional[float] = None
    pred_missing: int = 0
    gt_missing: int = 0
    type_mismatch: int = 0


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
    dependencies=[
        Depends(_debug_enabled),
        Depends(require_scopes(["debug:read_raw"])),
    ],
)
async def eval_batch_vs_gt(req: EvalBatchVsGTRequest) -> EvalBatchVsGTResponse:
    run_id = (req.run_id or "").strip()
    _validate_request_id_for_fs(run_id)

    if req.str_mode not in ("exact", "strip", "strip_collapse"):
        raise HTTPException(status_code=400, detail="str_mode must be one of: exact, strip, strip_collapse")

    batch_p = _find_batch_artifact_path(run_id, date=req.batch_date, max_days=req.max_days)
    if batch_p is None:
        raise HTTPException(status_code=404, detail="Batch artifact not found for given run_id (and batch_date/max_days).")

    batch_date = batch_p.parent.name if batch_p.parent else None
    batch_obj = _read_json_file(batch_p)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Batch artifact JSON is not an object.")

    task_id = batch_obj.get("task_id")
    task_snapshot = batch_obj.get("task_snapshot") if isinstance(batch_obj.get("task_snapshot"), dict) else {}
    schema_json = task_snapshot.get("schema_json") if isinstance(task_snapshot.get("schema_json"), dict) else None

    if schema_json is None:
        # fallback: use current task schema if available
        if isinstance(task_id, str) and task_id:
            schema_json = _get_task(task_id)["schema_value"]
        else:
            raise HTTPException(status_code=500, detail="Batch artifact does not contain task_snapshot.schema_json, and task_id is missing.")

    # Load GT
    gt_p = _validate_under_any_root(Path(req.gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])
    gt_obj = _read_json_file(gt_p)

    gt_index = _build_gt_index(gt_obj, image_key=req.gt_image_key)

    # Evaluation
    items = batch_obj.get("items") if isinstance(batch_obj.get("items"), list) else []
    if req.limit is not None:
        items = items[: int(req.limit)]

    # Precompute leaf paths (mostly for visibility)
    leaf_paths = _schema_leaf_paths(schema_json, schema_json)

    metrics: Dict[str, Dict[str, int]] = {}
    sample_summaries: List[EvalSampleSummary] = []

    gt_found = 0
    gt_missing = 0
    pred_found = 0
    pred_missing = 0

    for it in items:
        if not isinstance(it, dict):
            continue

        img_rel = it.get("image") or it.get("file") or ""
        img_base = os.path.basename(str(img_rel))
        request_id = str(it.get("request_id") or "")
        artifact_rel = it.get("artifact_rel") if isinstance(it.get("artifact_rel"), str) else None

        ok_flag = bool(it.get("ok"))
        schema_ok_flag = bool(it.get("schema_ok"))

        gt_rec = gt_index.get(img_base)

        if gt_rec is None:
            # fallback: sometimes GT stores relative path with folders that differ; try match by full rel as basename already done
            gt_missing += 1
        else:
            gt_found += 1
            if req.gt_record_key:
                sub = gt_rec.get(req.gt_record_key) if isinstance(gt_rec, dict) else None
                if isinstance(sub, dict):
                    gt_rec = sub

        # Read per-sample artifact (parsed prediction)
        art_obj: Optional[Dict[str, Any]] = None
        if artifact_rel:
            p = (ARTIFACTS_DIR / artifact_rel).resolve()
            try:
                p.relative_to(ARTIFACTS_DIR)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid artifact_rel in batch artifact (path traversal detected).")
            if p.exists():
                data = _read_json_file(p)
                if isinstance(data, dict):
                    art_obj = data
        if art_obj is None and request_id:
            # best-effort search by request_id across recent days
            p = _find_artifact_path(request_id, max_days=req.max_days)
            if p is not None and p.exists():
                data = _read_json_file(p)
                if isinstance(data, dict):
                    art_obj = data

        pred_value = _MISSING
        if isinstance(art_obj, dict):
            if "parsed" in art_obj:
                pred_value = art_obj.get("parsed")
            elif "result" in art_obj:
                pred_value = art_obj.get("result")

        if pred_value is _MISSING:
            pred_missing += 1
        else:
            pred_found += 1

        mismatches: List[Dict[str, Any]] = []
        notes: List[str] = []

        if gt_rec is None:
            notes.append("gt_not_found")
        else:
            # Compare only when GT exists; fields with missing GT values are tracked as gt_missing per field.
            _walk_schema(
                schema_json,
                pred_value,
                gt_rec,
                root_schema=schema_json,
                path=(),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=req.str_mode,
                key_fallbacks=req.key_fallbacks,
                mismatch_limit=int(req.mismatch_limit_total),
            )

        ss = EvalSampleSummary(
            image=str(img_rel),
            request_id=request_id,
            ok=ok_flag,
            schema_ok=schema_ok_flag,
            artifact_rel=artifact_rel,
            gt_found=gt_rec is not None,
            pred_found=pred_value is not _MISSING,
            mismatches=len(mismatches),
            mismatches_preview=mismatches[: int(req.mismatches_per_sample)],
            notes=notes,
        )
        sample_summaries.append(ss)

    # Build field metrics list
    field_rows: List[EvalFieldMetric] = []
    for path, m in metrics.items():
        total = int(m.get("total", 0))
        comparable = int(m.get("comparable", 0))
        matched = int(m.get("matched", 0))
        mismatched = int(m.get("mismatched", 0))
        acc = (matched / comparable) if comparable > 0 else None
        field_rows.append(
            EvalFieldMetric(
                path=path,
                total=total,
                comparable=comparable,
                matched=matched,
                mismatched=mismatched,
                accuracy=acc,
                pred_missing=int(m.get("pred_missing", 0)),
                gt_missing=int(m.get("gt_missing", 0)),
                type_mismatch=int(m.get("type_mismatch", 0)),
            )
        )

    # Sort: worst accuracy first (push fields with comparable==0 to the bottom)
    def _sort_key(x: EvalFieldMetric):
        uncmp = 1 if x.comparable <= 0 else 0
        acc = x.accuracy if x.accuracy is not None else 1.0
        return (uncmp, acc, -x.comparable, x.path)

    field_rows.sort(key=_sort_key)

    if req.sort_samples:
        # Sort by mismatches desc, then ok/schema_ok
        sample_summaries.sort(key=lambda s: (-s.mismatches, s.ok, s.schema_ok, s.image))
    
    comparable_total = sum(fr.comparable for fr in field_rows)
    matched_total = sum(fr.matched for fr in field_rows)
    overall_acc = (matched_total / comparable_total) if comparable_total > 0 else None


    eval_id = f"{run_id}__eval_{uuid.uuid4().hex[:8]}"
    
     # Save eval artifact for later inspection.
    
    eval_artifact_path = _save_eval_artifact(
        {
            "eval_id": eval_id,
            "run_id": run_id,
            "batch_date": batch_date,
            "task_id": task_id if isinstance(task_id, str) else None,
            "model": batch_obj.get("model") if isinstance(batch_obj.get("model"), str) else None,
            "inference_backend": (
                batch_obj.get("inference_backend")
                if isinstance(batch_obj.get("inference_backend"), str)
                else None
            ),
            "batch_summary": batch_obj.get("summary") if isinstance(batch_obj.get("summary"), dict) else {},
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
                "overall_accuracy": overall_acc,
            },
            "fields": [fr.model_dump() for fr in field_rows],
            "sample_summaries": [s.model_dump() for s in sample_summaries],
        }
    )
    return EvalBatchVsGTResponse(
        eval_id=eval_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        gt_path=req.gt_path,
        batch_artifact_path=str(batch_p),
        eval_artifact_path=str(eval_artifact_path),
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

    _save_artifact(
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
                art_p = _find_artifact_path(resp.request_id, max_days=30)
                artifact_rel = _to_artifact_rel(art_p) if art_p is not None else None

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

        # Reuse the same evaluation logic as the debug endpoint.
        eval_req = EvalBatchVsGTRequest(
            run_id=run_id,
            batch_date=Path(batch_artifact_path).parent.name,
            gt_path=req.gt_path,
            gt_image_key=req.gt_image_key or "image",
            sort_samples=False,
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
