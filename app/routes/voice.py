"""
Voice Routes – Twilio webhook handlers and WebSocket media stream.

Handles incoming calls via Twilio and manages real-time audio
streaming through Twilio Media Streams.
"""

import json
import base64
import logging
import asyncio
import audioop
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response

from app.services.call_orchestrator import call_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/incoming")
async def handle_incoming_call(request: Request):
    """
    Twilio webhook for incoming calls.
    Returns TwiML to connect the call to a WebSocket Media Stream.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    caller = form_data.get("From", "unknown")

    logger.info(f"Incoming call: {call_sid} from {caller}")

    # Determine the host for the WebSocket URL
    host = request.headers.get("host", "localhost:8000")
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"

    # Return TwiML that connects the call to our Media Stream WebSocket
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_scheme}://{host}/voice/stream">
            <Parameter name="call_sid" value="{call_sid}" />
            <Parameter name="caller" value="{caller}" />
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/stream")
async def websocket_media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.

    Receives audio from the caller, processes it through the
    STT → LLM → TTS pipeline, and streams audio back.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    call_sid = None
    caller = None
    stream_sid = None
    audio_buffer = bytearray()
    greeting_sent = False

    # Audio accumulation settings
    # Twilio sends 20ms chunks of μ-law 8kHz mono audio (160 bytes each)
    CHUNK_THRESHOLD = 3200   # ~0.4s of audio minimum before processing
    SILENCE_THRESHOLD = 30   # Consecutive silent chunks to detect pause (~600ms)
    SPEECH_RMS_THRESHOLD = 200  # RMS threshold for 16-bit PCM (after μ-law decode)

    silence_count = 0
    is_speaking = False
    is_processing = False  # Guard against re-entrant processing

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event_type = data.get("event")

            if event_type == "start":
                # Extract call metadata
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                custom_params = start_data.get("customParameters", {})
                call_sid = custom_params.get("call_sid", "unknown")
                caller = custom_params.get("caller", "unknown")

                logger.info(
                    f"Stream started: {stream_sid} | "
                    f"Call: {call_sid} | Caller: {caller}"
                )

                # Initialize call session and send greeting
                # Run in thread to avoid blocking the event loop
                greeting_audio_b64 = await asyncio.to_thread(
                    call_orchestrator.start_call, call_sid, caller
                )

                if greeting_audio_b64 and stream_sid:
                    is_processing = True  # Don't listen until greeting finishes playing
                    await _send_audio(websocket, stream_sid, greeting_audio_b64)
                    greeting_sent = True

            elif event_type == "media":
                # Skip audio accumulation while processing a response
                if is_processing:
                    continue

                payload = data.get("media", {}).get("payload", "")
                if payload:
                    chunk = base64.b64decode(payload)
                    audio_buffer.extend(chunk)

                    # Proper μ-law energy detection
                    try:
                        linear_chunk = audioop.ulaw2lin(chunk, 2)
                        rms_energy = audioop.rms(linear_chunk, 2)
                    except Exception:
                        rms_energy = 0

                    if rms_energy > SPEECH_RMS_THRESHOLD:
                        is_speaking = True
                        silence_count = 0
                    elif is_speaking:
                        silence_count += 1

                    # Process when we detect a pause after speech
                    if (
                        is_speaking
                        and silence_count >= SILENCE_THRESHOLD
                        and len(audio_buffer) > CHUNK_THRESHOLD
                    ):
                        audio_data = bytes(audio_buffer)
                        audio_buffer.clear()
                        is_speaking = False
                        silence_count = 0
                        is_processing = True

                        logger.info(
                            f"Processing {len(audio_data)} bytes of audio "
                            f"from {call_sid}"
                        )

                        # Run the BLOCKING STT → LLM → TTS pipeline in a
                        # thread so the WebSocket stays alive and responsive
                        try:
                            agent_text, response_audio_b64 = (
                                await asyncio.to_thread(
                                    call_orchestrator.process_customer_audio,
                                    call_sid,
                                    audio_data,
                                )
                            )

                            if response_audio_b64 and stream_sid:
                                await _send_audio(
                                    websocket, stream_sid, response_audio_b64
                                )
                                # Keep is_processing=True until the mark event
                                # confirms Twilio has finished playing the audio.
                                # The mark handler below resets it.
                            else:
                                is_processing = False  # No audio to play
                        except Exception as e:
                            logger.error(f"Pipeline error: {e}")
                            is_processing = False  # Reset immediately on error

            elif event_type == "stop":
                logger.info(f"Stream stopped: {stream_sid}")
                if call_sid:
                    call_orchestrator.end_call(call_sid)
                break

            elif event_type == "mark":
                mark_name = data.get("mark", {}).get("name")
                logger.debug(f"Mark received: {mark_name}")
                if mark_name == "response_end":
                    # Twilio has finished playing the agent's audio response.
                    # Now it's safe to start listening for the caller's reply.
                    is_processing = False
                    audio_buffer.clear()
                    is_speaking = False
                    silence_count = 0

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_sid}")
        if call_sid:
            call_orchestrator.end_call(call_sid)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if call_sid:
            call_orchestrator.end_call(call_sid)


async def _send_audio(websocket: WebSocket, stream_sid: str, audio_b64: str):
    """
    Send audio back to Twilio via the WebSocket Media Stream.
    """
    CHUNK_SIZE = 8000  # ~1 second of μ-law 8kHz audio

    audio_bytes = base64.b64decode(audio_b64)
    for i in range(0, len(audio_bytes), CHUNK_SIZE):
        chunk = audio_bytes[i : i + CHUNK_SIZE]
        chunk_b64 = base64.b64encode(chunk).decode("utf-8")

        media_message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": chunk_b64},
        }
        await websocket.send_text(json.dumps(media_message))

    # Mark event to track when audio finishes playing
    mark_message = {
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "response_end"},
    }
    await websocket.send_text(json.dumps(mark_message))


# ── REST endpoint for text-based testing ──────────────

@router.post("/chat")
async def text_chat(request: Request):
    """
    REST endpoint for testing the agent without voice.
    Send JSON: {"call_sid": "test-123", "message": "I'd like to book a haircut"}
    """
    body = await request.json()
    call_sid = body.get("call_sid", "test-session")
    message = body.get("message", "")

    if not message:
        return {"error": "No message provided"}

    agent_response, _ = call_orchestrator.process_customer_text(call_sid, message)

    return {
        "call_sid": call_sid,
        "agent_response": agent_response,
    }

