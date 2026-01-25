from pydantic_settings import BaseSettings, SettingsConfigDict

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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
