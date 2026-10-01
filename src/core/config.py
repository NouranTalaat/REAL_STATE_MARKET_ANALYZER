from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Central application configuration.

    Environment variables and .env values can override
    defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================
    # Application
    # ========================================================

    app_name: str = Field(
        default="Real Estate Market Intelligence API"
    )

    service_name: str = Field(
        default="real-estate-market-intelligence-api"
    )

    api_version: str = Field(
        default="2.0.0"
    )

    api_prefix: str = Field(
        default="/api/v2"
    )

    environment: str = Field(
        default="development"
    )

    debug: bool = Field(
        default=False
    )

    # ========================================================
    # Security
    # ========================================================

    cors_allowed_origins: str = Field(
        default="http://localhost,http://127.0.0.1"
    )

    allowed_hosts: str = Field(
        default="localhost,127.0.0.1"
    )

    max_request_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    @property
    def cors_origins(self) -> list[str]:
        """
        Convert comma-separated CORS origins into a list.
        """

        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def trusted_hosts(self) -> list[str]:
        """
        Convert comma-separated trusted hosts into a list.
        """

        hosts = [
            host.strip()
            for host in self.allowed_hosts.split(",")
            if host.strip()
        ]

        if self.environment.lower() in {
            "development",
            "testing",
            "test",
        }:
            if "testserver" not in hosts:
                hosts.append("testserver")

        return hosts

    @property
    def max_request_size_bytes(self) -> int:
        """
        Maximum accepted request body size in bytes.
        """

        return (
            self.max_request_size_mb
            * 1024
            * 1024
        )

    # ========================================================
    # Database
    # ========================================================

    database_server: str = Field(
        default="DESKTOP-C0V38AA"
    )

    database_name: str = Field(
        default="REAL_ESTATE_MARKET_INTELLIGENCE"
    )

    database_driver: str = Field(
        default="ODBC Driver 17 for SQL Server"
    )

    database_trusted_connection: bool = Field(
        default=True
    )

    database_trust_server_certificate: bool = Field(
        default=True
    )

    database_pool_pre_ping: bool = Field(
        default=True
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
    )

    database_max_overflow: int = Field(
        default=10,
        ge=0,
    )

    # ========================================================
    # ML Models
    # ========================================================

    models_dir: Path = Field(
        default=PROJECT_ROOT / "models"
    )

    sale_model_filename: str = Field(
        default="final_sale_model.pkl"
    )

    rent_model_filename: str = Field(
        default="final_rent_model.pkl"
    )

    # ========================================================
    # Runtime
    # ========================================================

    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )

    @property
    def sale_model_path(self) -> Path:
        return (
            self.models_dir
            / self.sale_model_filename
        )

    @property
    def rent_model_path(self) -> Path:
        return (
            self.models_dir
            / self.rent_model_filename
        )

    @property
    def database_url(self) -> str:
        """
        Build a SQL Server SQLAlchemy URL.
        """

        driver = self.database_driver.replace(
            " ",
            "+",
        )

        trusted_connection = (
            "yes"
            if self.database_trusted_connection
            else "no"
        )

        trust_server_certificate = (
            "yes"
            if self.database_trust_server_certificate
            else "no"
        )

        return (
            f"mssql+pyodbc://@"
            f"{self.database_server}/"
            f"{self.database_name}"
            f"?driver={driver}"
            f"&trusted_connection="
            f"{trusted_connection}"
            f"&TrustServerCertificate="
            f"{trust_server_certificate}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide cached settings instance.
    """

    return Settings()