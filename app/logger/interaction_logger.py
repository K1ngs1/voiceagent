"""
Interaction Logger – records all call interactions for QA and training.

Logs to a JSONL file for easy parsing and analysis.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Logs all call interactions to a JSONL file."""

    def __init__(self):
        self._log_path: Optional[Path] = None
        self._initialized = False

    def initialize(self):
        """Set up the log file path and ensure the directory exists."""
        settings = get_settings()
        self._log_path = Path(settings.log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info(f"Interaction logger initialized: {self._log_path}")

    def _ensure_initialized(self):
        """Lazy initialization."""
        if not self._initialized:
            self.initialize()

    def log_interaction(
        self,
        call_sid: str,
        customer_phone: Optional[str] = None,
        customer_transcript: Optional[str] = None,
        agent_response: Optional[str] = None,
        intent_detected: Optional[str] = None,
        tools_called: Optional[list[str]] = None,
        error: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        """
        Log a single interaction to the JSONL file.

        Args:
            call_sid: Twilio Call SID.
            customer_phone: Caller's phone number.
            customer_transcript: What the customer said (STT output).
            agent_response: What the agent said (LLM output).
            intent_detected: Detected intent (book, reschedule, cancel, inquiry).
            tools_called: List of tool names invoked.
            error: Error message if something went wrong.
            duration_seconds: Duration of this exchange.
            extra: Any additional metadata.
        """
        self._ensure_initialized()

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_sid": call_sid,
            "direction": "inbound",
            "customer_phone": customer_phone,
            "customer_transcript": customer_transcript,
            "agent_response": agent_response,
            "intent_detected": intent_detected,
            "tools_called": tools_called or [],
            "error": error,
            "duration_seconds": duration_seconds,
        }

        if extra:
            entry["extra"] = extra

        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write interaction log: {e}")

    def log_call_start(self, call_sid: str, customer_phone: Optional[str] = None):
        """Log the start of a new call."""
        self.log_interaction(
            call_sid=call_sid,
            customer_phone=customer_phone,
            extra={"event": "call_started"},
        )

    def log_call_end(
        self,
        call_sid: str,
        duration_seconds: Optional[float] = None,
        summary: Optional[str] = None,
    ):
        """Log the end of a call."""
        self.log_interaction(
            call_sid=call_sid,
            duration_seconds=duration_seconds,
            extra={"event": "call_ended", "summary": summary},
        )

    def log_error(self, call_sid: str, error: str, context: Optional[str] = None):
        """Log an error during a call."""
        self.log_interaction(
            call_sid=call_sid,
            error=error,
            extra={"event": "error", "context": context},
        )


# Singleton instance
interaction_logger = InteractionLogger()
