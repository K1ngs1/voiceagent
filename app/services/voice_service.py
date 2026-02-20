"""
Voice Service – ElevenLabs TTS & STT.

Converts text to speech (for the agent's voice) and
speech to text (for transcribing the customer).
"""

import logging
import io
import base64
from typing import Optional

from elevenlabs.client import ElevenLabs
from elevenlabs import play  # noqa – available for local testing

from app.config import get_settings

logger = logging.getLogger(__name__)


class VoiceService:
    """ElevenLabs-powered voice service for TTS and STT."""

    def __init__(self):
        self._client: Optional[ElevenLabs] = None
        self._settings = None
        self._voice_id: Optional[str] = None

    def initialize(self):
        """Set up the ElevenLabs client."""
        self._settings = get_settings()
        self._client = ElevenLabs(api_key=self._settings.elevenlabs_api_key)
        self._voice_id = self._settings.elevenlabs_voice_id
        logger.info("ElevenLabs voice service initialized.")

    def _get_client(self) -> ElevenLabs:
        """Lazy-initialize and return the ElevenLabs client."""
        if self._client is None:
            self.initialize()
        return self._client

    def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_turbo_v2",
        output_format: str = "ulaw_8000",
    ) -> bytes:
        """
        Convert text to spoken audio using ElevenLabs TTS.

        Args:
            text: The text to synthesize.
            voice_id: Override voice ID (defaults to configured voice).
            model_id: ElevenLabs model to use ('eleven_turbo_v2' for low latency).
            output_format: Audio format. 'ulaw_8000' for Twilio compatibility.

        Returns:
            Raw audio bytes.
        """
        client = self._get_client()
        vid = voice_id or self._voice_id

        try:
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=vid,
                model_id=model_id,
                output_format=output_format,
            )

            # Collect all audio chunks
            audio_bytes = b""
            for chunk in audio_generator:
                audio_bytes += chunk

            logger.info(f"TTS generated: {len(audio_bytes)} bytes for {len(text)} chars")
            return audio_bytes

        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise

    def text_to_speech_base64(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> str:
        """
        Convert text to speech and return as base64-encoded string.
        Useful for sending audio over WebSocket to Twilio.

        Args:
            text: Text to synthesize.
            voice_id: Override voice ID.

        Returns:
            Base64-encoded audio string.
        """
        audio_bytes = self.text_to_speech(text, voice_id)
        return base64.b64encode(audio_bytes).decode("utf-8")

    def speech_to_text(
        self,
        audio_bytes: bytes,
        model_id: str = "scribe_v1",
    ) -> str:
        """
        Transcribe spoken audio to text using ElevenLabs STT.

        Args:
            audio_bytes: Raw μ-law 8kHz audio bytes from Twilio.
            model_id: ElevenLabs STT model.

        Returns:
            Transcribed text string.
        """
        import audioop
        import wave

        client = self._get_client()

        try:
            # Convert raw μ-law 8kHz mono to a proper WAV file
            # ElevenLabs STT needs a real audio file, not raw bytes
            linear_pcm = audioop.ulaw2lin(audio_bytes, 2)  # 16-bit PCM

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)       # Mono
                wf.setsampwidth(2)       # 16-bit
                wf.setframerate(8000)    # 8kHz (Twilio's rate)
                wf.writeframes(linear_pcm)

            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            result = client.speech_to_text.convert(
                file=wav_buffer,
                model_id=model_id,
            )

            transcript = result.text if hasattr(result, "text") else str(result)
            logger.info(f"STT transcribed: '{transcript[:80]}...'")
            return transcript

        except Exception as e:
            logger.error(f"ElevenLabs STT error: {e}")
            raise

    def text_to_speech_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_turbo_v2",
        output_format: str = "ulaw_8000",
    ):
        """
        Stream TTS audio chunks for lower latency.

        Args:
            text: Text to synthesize.
            voice_id: Override voice ID.
            model_id: ElevenLabs model.
            output_format: Audio format.

        Yields:
            Audio byte chunks.
        """
        client = self._get_client()
        vid = voice_id or self._voice_id

        try:
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=vid,
                model_id=model_id,
                output_format=output_format,
            )

            for chunk in audio_generator:
                yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs TTS streaming error: {e}")
            raise


# Singleton instance
voice_service = VoiceService()
