"""
Tests for the LLM Agent.

Uses mocks – no real OpenRouter API calls are made.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.llm_agent import LLMAgent


@pytest.fixture
def mock_settings():
    """Mock application settings."""
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


@pytest.fixture
def agent(mock_settings):
    """Create an LLM agent with mocked dependencies."""
    with patch("app.services.llm_agent.get_settings", return_value=mock_settings), \
         patch("app.prompts.salon_agent.get_settings", return_value=mock_settings), \
         patch("app.config.get_settings", return_value=mock_settings), \
         patch("app.services.llm_agent.interaction_logger") as mock_logger:
        agent = LLMAgent()
        agent._client = MagicMock()
        agent._model = "openai/gpt-4o"
        agent._settings = mock_settings
        yield agent


def _make_mock_response(content: str, tool_calls=None):
    """Helper to create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    choice.message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }

    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMChat:
    """Test the LLM chat method."""

    def test_simple_response(self, agent):
        """Should return text response for a simple query."""
        mock_resp = _make_mock_response(
            "Welcome! How can I help you today?"
        )
        agent._client.chat.completions.create.return_value = mock_resp

        text, history, tools = agent.chat([], "Hello!", "test-sid")

        assert "Welcome" in text or "help" in text.lower()
        assert len(history) == 2  # user + assistant
        assert len(tools) == 0

    def test_tool_call_flow(self, agent):
        """Should handle tool calls correctly."""
        # First response: LLM wants to call a tool
        tool_call = MagicMock()
        tool_call.id = "tc_1"
        tool_call.function.name = "search_knowledge_base"
        tool_call.function.arguments = json.dumps({"query": "haircut price"})

        first_resp = MagicMock()
        first_choice = MagicMock()
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.content = None
        first_choice.message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": json.dumps({"query": "haircut price"}),
                    },
                }
            ],
        }
        first_resp.choices = [first_choice]

        # Second response: LLM gives final answer after tool result
        second_resp = _make_mock_response(
            "A women's haircut is $85 and takes about an hour."
        )

        agent._client.chat.completions.create.side_effect = [
            first_resp,
            second_resp,
        ]

        # Mock RAG service
        with patch("app.services.llm_agent.rag_service") as mock_rag:
            mock_rag.query.return_value = [
                {
                    "content": "Women's Haircut & Style: $85, 60 minutes",
                    "source": "services",
                    "relevance_score": 0.9,
                }
            ]

            text, history, tools = agent.chat(
                [], "How much is a haircut?", "test-sid"
            )

        assert "$85" in text
        assert "search_knowledge_base" in tools

    def test_conversation_history_maintained(self, agent):
        """Should maintain conversation history across turns."""
        resp1 = _make_mock_response("I'd love to help! What service?")
        resp2 = _make_mock_response("Great, a haircut! Any preferred stylist?")

        agent._client.chat.completions.create.side_effect = [resp1, resp2]

        _, history1, _ = agent.chat([], "I want to book an appointment", "test-sid")
        assert len(history1) == 2

        _, history2, _ = agent.chat(history1, "A haircut please", "test-sid")
        assert len(history2) == 4  # 2 from first turn + 2 from second

    def test_api_error_handling(self, agent):
        """Should return a fallback message on API error."""
        agent._client.chat.completions.create.side_effect = Exception("API timeout")

        text, history, tools = agent.chat([], "Hello", "test-sid")

        assert "sorry" in text.lower() or "trouble" in text.lower()


class TestGreeting:
    """Test greeting generation."""

    def test_greeting_includes_salon_name(self, agent):
        """Greeting should include the salon name."""
        with patch("app.services.llm_agent.get_settings") as mock_get:
            mock_get.return_value = agent._settings
            greeting = agent.get_greeting()

        assert "Test Salon" in greeting
