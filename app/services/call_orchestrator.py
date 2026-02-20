"""
Call Orchestrator – manages the conversation flow for each call.

Ties together Voice (ElevenLabs), LLM (OpenRouter), Calendar,
and RAG services into a coherent per-call conversation loop.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from app.models.schemas import CallSession
from app.services.llm_agent import llm_agent
from app.services.voice_service import voice_service
from app.logger.interaction_logger import interaction_logger

logger = logging.getLogger(__name__)


class CallOrchestrator:
    """Manages the lifecycle of an inbound phone call."""

    def __init__(self):
        self._active_sessions: dict[str, CallSession] = {}

    def start_call(self, call_sid: str, customer_phone: Optional[str] = None) -> str:
        """
        Initialize a new call session and return the greeting audio (base64).

        Args:
            call_sid: Twilio Call SID.
            customer_phone: Caller's phone number.

        Returns:
            Base64-encoded greeting audio.
        """
        # Create session
        session = CallSession(
            call_sid=call_sid,
            customer_phone=customer_phone,
            started_at=datetime.now(timezone.utc),
        )
        self._active_sessions[call_sid] = session

        # Log call start
        interaction_logger.log_call_start(call_sid, customer_phone)

        # Generate greeting
        greeting_text = llm_agent.get_greeting()
        logger.info(f"Call started: {call_sid} | Greeting: {greeting_text}")

        # Convert greeting to speech
        try:
            greeting_audio_b64 = voice_service.text_to_speech_base64(greeting_text)
        except Exception as e:
            logger.error(f"TTS error for greeting: {e}")
            greeting_audio_b64 = ""

        # Store greeting in conversation history
        session.conversation_history.append(
            {"role": "assistant", "content": greeting_text}
        )

        return greeting_audio_b64

    def process_customer_audio(
        self, call_sid: str, audio_bytes: bytes
    ) -> tuple[str, str]:
        """
        Process incoming customer audio: STT → LLM → TTS.

        Args:
            call_sid: Twilio Call SID.
            audio_bytes: Raw audio bytes from the customer.

        Returns:
            Tuple of (agent_response_text, base64_audio_response).
        """
        session = self._active_sessions.get(call_sid)
        if session is None:
            logger.warning(f"No active session for call {call_sid}")
            return "", ""

        start_time = time.time()

        # ── Step 1: Speech-to-Text ─────────────────────
        try:
            customer_text = voice_service.speech_to_text(audio_bytes)
        except Exception as e:
            logger.error(f"STT error: {e}")
            interaction_logger.log_error(call_sid, str(e), "stt")
            error_response = "I'm sorry, I didn't catch that. Could you say that again?"
            error_audio = voice_service.text_to_speech_base64(error_response)
            return error_response, error_audio

        if not customer_text or customer_text.strip() == "":
            return "", ""

        logger.info(f"Customer ({call_sid}): {customer_text}")

        # ── Step 2: LLM Processing ────────────────────
        try:
            agent_response, updated_history, tools_called = llm_agent.chat(
                conversation_history=session.conversation_history,
                user_message=customer_text,
                call_sid=call_sid,
            )
            session.conversation_history = updated_history
        except Exception as e:
            logger.error(f"LLM error: {e}")
            interaction_logger.log_error(call_sid, str(e), "llm")
            error_response = (
                "I apologize, I'm experiencing a brief issue. "
                "Could you repeat what you said?"
            )
            error_audio = voice_service.text_to_speech_base64(error_response)
            return error_response, error_audio

        logger.info(f"Agent ({call_sid}): {agent_response}")

        # ── Step 3: Text-to-Speech ────────────────────
        try:
            response_audio_b64 = voice_service.text_to_speech_base64(agent_response)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            interaction_logger.log_error(call_sid, str(e), "tts")
            response_audio_b64 = ""

        elapsed = time.time() - start_time
        logger.info(f"Turn processed in {elapsed:.2f}s | Tools: {tools_called}")

        return agent_response, response_audio_b64

    def process_customer_text(self, call_sid: str, text: str) -> tuple[str, str]:
        """
        Process a text input directly (useful for testing without voice).

        Args:
            call_sid: Call or session identifier.
            text: Customer's text message.

        Returns:
            Tuple of (agent_response_text, base64_audio_response).
        """
        session = self._active_sessions.get(call_sid)
        if session is None:
            # Create a temporary session
            session = CallSession(
                call_sid=call_sid,
                started_at=datetime.now(timezone.utc),
            )
            self._active_sessions[call_sid] = session

        # LLM processing
        agent_response, updated_history, tools_called = llm_agent.chat(
            conversation_history=session.conversation_history,
            user_message=text,
            call_sid=call_sid,
        )
        session.conversation_history = updated_history

        # TTS
        try:
            response_audio_b64 = voice_service.text_to_speech_base64(agent_response)
        except Exception:
            response_audio_b64 = ""

        return agent_response, response_audio_b64

    def end_call(self, call_sid: str):
        """
        Clean up a call session.

        Args:
            call_sid: Twilio Call SID.
        """
        session = self._active_sessions.pop(call_sid, None)
        if session:
            session.is_active = False
            duration = (
                datetime.now(timezone.utc) - session.started_at
            ).total_seconds()
            interaction_logger.log_call_end(
                call_sid=call_sid,
                duration_seconds=duration,
                summary=f"Call lasted {duration:.0f}s with {len(session.conversation_history)} exchanges",
            )
            logger.info(
                f"Call ended: {call_sid} | Duration: {duration:.0f}s | "
                f"Exchanges: {len(session.conversation_history)}"
            )

    def get_session(self, call_sid: str) -> Optional[CallSession]:
        """Get an active call session."""
        return self._active_sessions.get(call_sid)

    def get_active_call_count(self) -> int:
        """Get the number of active calls."""
        return len(self._active_sessions)


# Singleton instance
call_orchestrator = CallOrchestrator()
