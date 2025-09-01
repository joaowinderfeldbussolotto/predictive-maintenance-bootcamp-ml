from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Model settings
    MODEL_DIR: str = "ml_models"
    PREDICTION_THRESHOLD: float = 0.5
    MAX_BATCH_SIZE: int = 1000

    model_config = SettingsConfigDict(env_file="api/.env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
