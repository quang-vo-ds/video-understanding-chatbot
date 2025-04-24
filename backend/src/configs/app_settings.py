from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    LLM_PROVIDER: str
    LLM_URL: str
    LLM_NAME: str

    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_URL: str

    LOGGER: str

settings = AppSettings()