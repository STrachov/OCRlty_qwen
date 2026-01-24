from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    AUTH_ENABLED: bool = True
    AUTH_DB_PATH: str = "/workspace/auth/auth.db"
    API_KEY_PEPPER: str

    ARTIFACTS_DIR: str = "/workspace/artifacts"
    VLLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
