"""Centralised configuration loaded from environment / .env file.

Mirrors patternalarm's Config.scala pattern: defaults with env-var overrides.
Uses pydantic-settings for validation (ransom-rampage convention).
"""

from pydantic_settings import BaseSettings


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = "localhost:9092"
    api_key: str = ""
    api_secret: str = ""
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "PLAIN"

    topic_raw: str = "cpv-raw"
    topic_hardcoded: str = "cpv-hardcoded"
    topic_sparkml: str = "cpv-sparkml"
    topic_deeplearning: str = "cpv-deeplearning"
    topic_solver: str = "cpv-solver"
    topic_langchain: str = "cpv-langchain"
    topic_n8n: str = "cpv-n8n"
    topic_mcp: str = "cpv-mcp"

    group_hardcoded: str = "spendlabel-hardcoded"
    group_sparkml: str = "spendlabel-sparkml"
    group_deeplearning: str = "spendlabel-deeplearning"
    group_solver: str = "spendlabel-solver"
    group_langchain: str = "spendlabel-langchain"
    group_n8n: str = "spendlabel-n8n"
    group_mcp: str = "spendlabel-mcp"

    model_config = {"env_prefix": "KAFKA_"}


class FlinkSettings(BaseSettings):
    checkpointing_interval_ms: int = 60_000

    model_config = {"env_prefix": "FLINK_"}


class Settings(BaseSettings):
    kafka: KafkaSettings = KafkaSettings()
    flink: FlinkSettings = FlinkSettings()

    model_config = {"env_file": "config/.env", "env_file_encoding": "utf-8"}


settings = Settings()
