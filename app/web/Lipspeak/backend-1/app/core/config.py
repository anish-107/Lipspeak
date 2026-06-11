"""config.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Centralized application configuration.
@date: 11 June 2026
@returns: Application settings instance.

"""


# Imports
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# Settings
class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str
    APP_ENV: str

    # Database
    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_DATABASE: str
    MYSQL_USER: str
    MYSQL_PASSWORD: str

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    JWT_EXPIRE_MINUTES: int

    # AWS
    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET: str

    # GRID
    GRID_GRPC_HOST: str
    GRID_GRPC_PORT: int

    # AVSR
    AVSR_GRPC_HOST: str
    AVSR_GRPC_PORT: int

    model_config = (
        SettingsConfigDict(
            env_file=".env",
            extra="ignore",
        )
    )


settings = Settings()  # type: ignore