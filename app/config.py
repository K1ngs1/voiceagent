"""
Configuration management for the Salon AI Voice Agent.
Loads all API keys and settings from environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── OpenRouter ─────────────────────────────────────
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_model: str = Field(
        default="openai/gpt-4o", description="LLM model to use via OpenRouter"
    )

    # ── ElevenLabs ─────────────────────────────────────
    elevenlabs_api_key: str = Field(..., description="ElevenLabs API key")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        description="ElevenLabs voice ID for TTS",
    )

    # ── Twilio ─────────────────────────────────────────
    twilio_account_sid: str = Field(..., description="Twilio Account SID")
    twilio_auth_token: str = Field(..., description="Twilio Auth Token")
    twilio_phone_number: str = Field(..., description="Twilio phone number")

    # ── Google Calendar ────────────────────────────────
    google_calendar_id: str = Field(..., description="Google Calendar ID")
    google_service_account_file: str = Field(
        default="credentials/service_account.json",
        description="Path to Google service account JSON",
    )

    # ── Salon Configuration ────────────────────────────
    salon_name: str = Field(default="Luxe Beauty Salon", description="Salon name")
    salon_timezone: str = Field(
        default="America/Los_Angeles", description="Salon timezone"
    )
    salon_phone: str = Field(default="+1234567890", description="Salon phone number")

    # ── Logging ────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(
        default="logs/interactions.jsonl", description="Path to interaction log file"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
