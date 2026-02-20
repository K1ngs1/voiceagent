"""
Shared pytest fixtures for the Salon AI Voice Agent test suite.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_settings():
    """
    Mock application settings shared across all test modules.
    Provides sensible defaults for all config fields without
    requiring a real .env file.
    """
    settings = MagicMock()
    settings.openrouter_api_key = "test-key"
    settings.openrouter_model = "openai/gpt-4o"
    settings.salon_name = "Test Salon"
    settings.salon_timezone = "America/Los_Angeles"
    settings.salon_phone = "+15551234567"
    settings.elevenlabs_api_key = "test-key"
    settings.elevenlabs_voice_id = "test-voice"
    settings.twilio_account_sid = "test"
    settings.twilio_auth_token = "test"
    settings.twilio_phone_number = "+15559999999"
    settings.google_calendar_id = "test@calendar.google.com"
    settings.google_service_account_file = "test.json"
    settings.log_level = "INFO"
    settings.log_file = "/tmp/test_interactions.jsonl"
    return settings
