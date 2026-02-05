from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator

class Settings(BaseSettings):
    # --- Auth ---
    AUTH_ENABLED: bool = True
    AUTH_DB_PATH: str = "/workspace/auth/auth.db"
    API_KEY_PEPPER: str

    # --- Debug / local dev ---
    # If False, all debug features/endpoints are disabled, even if a key has debug scopes.
    DEBUG_MODE: bool = False

    # Switch inference backend without changing code.
    # - "vllm": call the vLLM OpenAI-compatible API
    # - "mock": return deterministic fake responses (for local testing without GPU/model)
    INFERENCE_BACKEND: str = "vllm"

    # Hard caps (apply to both vLLM and mock backends)
    MAX_PROMPT_CHARS: int = 20000
    MAX_TOKENS_CAP: int = 256

    # --- App ---
    ARTIFACTS_DIR: str = "/workspace/artifacts"
    VLLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"



    # --- Object storage (S3 / Cloudflare R2) ---
    # Set S3_BUCKET to enable storing artifacts in S3-compatible storage.
    # For Cloudflare R2, also set:
    #   S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    #   S3_REGION=auto
    #
    # Credentials: boto3 will read AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from env.
    S3_BUCKET: str = ""
    S3_ENDPOINT_URL: str | None = None
    S3_REGION: str = "auto"
    # Support both S3_PREFIX and legacy S3_ARTIFACTS_PREFIX env var names.
    S3_PREFIX: str = Field(default="ocrlty", validation_alias=AliasChoices("S3_PREFIX", "S3_ARTIFACTS_PREFIX"))
    S3_ALLOW_OVERWRITE: bool = False
    S3_PRESIGN_TTL_S: int = 3600
    S3_FORCE_PATH_STYLE: bool = False

    @field_validator("S3_BUCKET", mode="before")
    @classmethod
    def _strip_s3_bucket(cls, v):
        return str(v or "").strip()

    @field_validator("S3_ENDPOINT_URL", mode="before")
    @classmethod
    def _strip_s3_endpoint(cls, v):
        s = str(v or "").strip()
        return s or None

    @field_validator("S3_REGION", mode="before")
    @classmethod
    def _strip_s3_region(cls, v):
        return str(v or "auto").strip()

    @field_validator("S3_PREFIX", mode="before")
    @classmethod
    def _strip_s3_prefix(cls, v):
        return str(v or "ocrlty").strip().strip("/")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
