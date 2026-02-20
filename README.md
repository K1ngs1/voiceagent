# 🏪 Salon AI Voice Agent

An AI-powered voice agent that handles inbound phone calls for a salon. It can book, reschedule, and cancel appointments, and answer questions about services, pricing, stylists, and policies — all through natural voice conversation.

## Architecture

```
📞 Incoming Call → Twilio → FastAPI Server
                              ├── ElevenLabs STT (speech → text)
                              ├── OpenRouter LLM (GPT-4o – conversation + tool calling)
                              │     ├── Google Calendar API (availability, CRUD)
                              │     └── RAG Knowledge Base (services, prices, policies)
                              └── ElevenLabs TTS (text → speech) → Twilio → Caller
```

## Features

- **Natural voice conversation** via ElevenLabs TTS/STT
- **Intelligent intent detection** – booking, rescheduling, cancellation, inquiries
- **Google Calendar integration** – real-time availability and appointment management
- **RAG knowledge base** – accurate answers about services, pricing, and policies
- **Tool calling** – LLM autonomously queries calendar and knowledge base
- **Conversation context** – maintains full session state per call
- **Interaction logging** – JSONL logs for QA and training

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/aryanpandey/Desktop/vc
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required API keys:**
| Service | Get it at |
|---------|-----------|
| OpenRouter | https://openrouter.ai/keys |
| ElevenLabs | https://elevenlabs.io/app/settings/api-keys |
| Twilio | https://console.twilio.com |
| Google Calendar | See below |

### 3. Set Up Google Calendar

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project and enable the **Google Calendar API**
3. Create a **Service Account** under IAM & Admin
4. Download the JSON key file → save as `credentials/service_account.json`
5. Open Google Calendar → Settings → Share your calendar with the **service account email** (grant "Make changes to events")
6. Copy your **Calendar ID** → paste into `.env` as `GOOGLE_CALENDAR_ID`

### 4. Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Connect Twilio (for real calls)

```bash
# In another terminal:
ngrok http 8000
```

Then in the [Twilio Console](https://console.twilio.com):
- Go to your phone number → Voice Configuration
- Set webhook URL to: `https://YOUR_NGROK_URL/voice/incoming`

### 6. Test Without Voice

You can test the agent via the REST API without needing Twilio/voice:

```bash
curl -X POST http://localhost:8000/voice/chat \
  -H "Content-Type: application/json" \
  -d '{"call_sid": "test-1", "message": "Hi, I would like to book a haircut"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + service info |
| `/health` | GET | Health status |
| `/voice/incoming` | POST | Twilio webhook for calls |
| `/voice/stream` | WS | WebSocket for audio streaming |
| `/voice/chat` | POST | Text-based testing endpoint |
| `/docs` | GET | OpenAPI documentation |

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_rag_service.py -v
python -m pytest tests/test_calendar_service.py -v
python -m pytest tests/test_llm_agent.py -v
```

## Project Structure

```
vc/
├── app/
│   ├── main.py                  # FastAPI app entry
│   ├── config.py                # Settings from .env
│   ├── routes/voice.py          # Twilio webhooks + WebSocket
│   ├── services/
│   │   ├── llm_agent.py         # OpenRouter LLM + tool calling
│   │   ├── voice_service.py     # ElevenLabs TTS/STT
│   │   ├── calendar_service.py  # Google Calendar CRUD
│   │   ├── rag_service.py       # ChromaDB knowledge retrieval
│   │   └── call_orchestrator.py # Per-call session + pipeline
│   ├── models/schemas.py        # Pydantic data models
│   ├── prompts/salon_agent.py   # System prompt + tool schemas
│   └── logger/interaction_logger.py
├── knowledge_base/salon_data.json
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Salon Knowledge Base

The RAG knowledge base (`knowledge_base/salon_data.json`) includes:
- **19 services** with prices and durations
- **5 stylists** with specialties and availability
- **6 policies** (cancellation, late arrival, deposits, etc.)
- **10 FAQs**
- **Location info** with hours and parking

Edit `salon_data.json` to customize for your salon.
