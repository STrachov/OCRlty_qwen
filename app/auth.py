# app/auth.py
import hmac
import hashlib
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from fastapi import HTTPException, Request
from app.settings import settings

# --- Env / config ---
AUTH_ENABLED = settings.AUTH_ENABLED
AUTH_DB_PATH = settings.AUTH_DB_PATH
API_KEY_PEPPER = settings.API_KEY_PEPPER.strip()

ROLE_PRESET_SCOPES: Dict[str, List[str]] = {
    "client": ["extract:run"],
    "debugger": ["extract:run", "debug:run"],
    "admin": ["extract:run", "debug:run", "debug:read_raw"],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --- Model ---

@dataclass(frozen=True)
class ApiPrincipal:
    api_key_id: int
    key_id: str
    role: str
    scopes: Set[str]


# --- Hashing ---

def _require_pepper() -> str:
    if not API_KEY_PEPPER:
        raise RuntimeError("API_KEY_PEPPER is required (set env var) when auth is enabled.")
    return API_KEY_PEPPER


def compute_key_hash(api_key: str, pepper: str) -> str:
    # HMAC-SHA256(secret=pepper, msg=api_key)
    return hmac.new(
        pepper.encode("utf-8"),
        api_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# --- SQLite helpers ---

def _ensure_db_dir(db_path: str) -> None:
    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    _ensure_db_dir(db_path)
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db(db_path: str = AUTH_DB_PATH) -> None:
    """
    Creates tables if not exist.
    Safe to call on every startup.
    """
    if AUTH_ENABLED:
        _require_pepper()

    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                api_key_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id       TEXT NOT NULL UNIQUE,
                key_hash     TEXT NOT NULL UNIQUE,
                role         TEXT NOT NULL,
                scopes       TEXT NOT NULL, -- JSON array or CSV
                is_active    INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT NOT NULL,
                last_used_at TEXT,
                revoked_at   TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_keyid ON api_keys(key_id);")
        conn.commit()


# --- Key parsing ---

def extract_api_key_from_request(request: Request) -> Optional[str]:
    """
    Priority:
      1) Authorization: Bearer <key>
      2) X-API-Key: <key>
    """
    auth = request.headers.get("authorization")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None
        # If Authorization exists but not Bearer, treat as invalid.
        raise HTTPException(status_code=401, detail="Invalid Authorization header. Use 'Bearer <api_key>'.")

    xkey = request.headers.get("x-api-key")
    if xkey:
        return xkey.strip() or None

    return None


# --- Lookup / dependency ---

def lookup_principal(api_key: str, db_path: str = AUTH_DB_PATH) -> Optional[ApiPrincipal]:
    pepper = _require_pepper()
    kh = compute_key_hash(api_key, pepper)

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT api_key_id, key_id, role, scopes, is_active
            FROM api_keys
            WHERE key_hash = ?
            LIMIT 1;
            """,
            (kh,),
        ).fetchone()

        if row is None:
            return None
        if int(row["is_active"]) != 1:
            return None

        # Update last_used_at (best-effort)
        try:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE api_key_id = ?;",
                (_utc_now_iso(), int(row["api_key_id"])),
            )
            conn.commit()
        except Exception:
            pass

        scopes_raw = str(row["scopes"] or "").strip()
        scopes_set: Set[str] = set()
        if scopes_raw:
            try:
                parsed = json.loads(scopes_raw)
                if isinstance(parsed, list):
                    scopes_set = {str(s).strip() for s in parsed if str(s).strip()}
                else:
                    # fallback to CSV
                    scopes_set = {s.strip() for s in scopes_raw.split(",") if s.strip()}
            except Exception:
                scopes_set = {s.strip() for s in scopes_raw.split(",") if s.strip()}

        return ApiPrincipal(
            api_key_id=int(row["api_key_id"]),
            key_id=str(row["key_id"]),
            role=str(row["role"]),
            scopes=scopes_set,
        )


def require_api_key(request: Request) -> ApiPrincipal:
    """
    FastAPI dependency:
      - If AUTH_ENABLED=0 -> returns a synthetic principal (for local/dev), without checks.
      - If AUTH_ENABLED=1 -> requires valid active key.
    """
    if not AUTH_ENABLED:
        # Dev mode: keep API open, but still provide a principal shape.
        return ApiPrincipal(api_key_id=0, key_id="anonymous", role="anonymous", scopes=set())

    api_key = extract_api_key_from_request(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key (Authorization: Bearer ... or X-API-Key).")

    principal = lookup_principal(api_key)
    if principal is None:
        raise HTTPException(status_code=403, detail="Invalid or inactive API key.")

    # Store on request for logs/debug if needed (do NOT store the raw api key).
    request.state.principal = principal
    return principal


# --- CLI utilities (used by app/auth_cli.py) ---

def role_default_scopes(role: str) -> List[str]:
    return ROLE_PRESET_SCOPES.get(role, [])


def create_api_key(
    key_id: str,
    role: str,
    scopes: Optional[Sequence[str]] = None,
    db_path: str = AUTH_DB_PATH,
) -> Tuple[str, ApiPrincipal]:
    """
    Creates a new API key. Returns (raw_api_key, principal).
    Raw key is shown once to the operator.
    """
    pepper = _require_pepper()
    init_auth_db(db_path)

    role = role.strip()
    if role not in ROLE_PRESET_SCOPES:
        raise ValueError(f"Unknown role: {role}. Allowed: {sorted(ROLE_PRESET_SCOPES.keys())}")

    if scopes is None:
        scopes_list = role_default_scopes(role)
    else:
        scopes_list = [s.strip() for s in scopes if s and s.strip()]

    raw_key = secrets.token_urlsafe(32)  # ~43 chars
    kh = compute_key_hash(raw_key, pepper)

    with _connect(db_path) as conn:
        created_at = _utc_now_iso()
        scopes_json = json.dumps(scopes_list, ensure_ascii=False)

        try:
            cur = conn.execute(
                """
                INSERT INTO api_keys (key_id, key_hash, role, scopes, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?);
                """,
                (key_id, kh, role, scopes_json, created_at),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Failed to create key. Probably duplicate key_id. Details: {e}") from e

        api_key_id = int(cur.lastrowid)

    principal = ApiPrincipal(api_key_id=api_key_id, key_id=key_id, role=role, scopes=set(scopes_list))
    return raw_key, principal


def revoke_api_key(key_id: str, db_path: str = AUTH_DB_PATH) -> bool:
    init_auth_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE api_keys
            SET is_active = 0, revoked_at = ?
            WHERE key_id = ? AND is_active = 1;
            """,
            (_utc_now_iso(), key_id),
        )
        conn.commit()
        return cur.rowcount > 0


def list_api_keys(db_path: str = AUTH_DB_PATH) -> List[Dict[str, Any]]:
    init_auth_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT api_key_id, key_id, role, scopes, is_active, created_at, last_used_at, revoked_at
            FROM api_keys
            ORDER BY api_key_id ASC;
            """
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "api_key_id": int(r["api_key_id"]),
                "key_id": str(r["key_id"]),
                "role": str(r["role"]),
                "scopes": str(r["scopes"]),
                "is_active": bool(int(r["is_active"])),
                "created_at": r["created_at"],
                "last_used_at": r["last_used_at"],
                "revoked_at": r["revoked_at"],
            }
        )
    return out
