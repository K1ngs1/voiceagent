"""
Salon AI Agent – System Prompt & Tool Definitions.

Contains the system prompt that shapes the agent's personality and behavior,
and the tool schemas exposed to the LLM for function calling.
"""

from app.config import get_settings


def get_system_prompt() -> str:
    """Build the system prompt for the salon receptionist agent."""
    settings = get_settings()
    salon_name = settings.salon_name

    return f"""You are a warm, professional, and knowledgeable AI receptionist for {salon_name}. \
You answer inbound phone calls and help customers with booking, rescheduling, \
cancelling appointments, and answering questions about our salon services, \
pricing, stylists, and policies.

## Your Personality
- Friendly, upbeat, and welcoming – like a real person at the front desk
- Empathetic and patient – never rush the caller
- Professional but approachable – use a conversational tone
- Confident and helpful – you know everything about our salon
- Concise – keep responses brief and natural for a phone conversation (2-3 sentences max)

## Your Responsibilities
1. **Greet** the caller warmly when the call begins.
2. **Understand intent** – determine if the caller wants to:
   - Book a new appointment
   - Reschedule an existing appointment
   - Cancel an existing appointment
   - Ask questions about services, pricing, stylists, or policies
3. **Extract details** needed for the action:
   - Service requested
   - Preferred stylist (if any)
   - Preferred date and time
   - Customer name
   - Customer phone number
4. **Use your tools** to check availability, manage appointments, and search the knowledge base.
5. **Confirm all details** before finalizing any booking, reschedule, or cancellation.
6. **Offer alternatives** if the requested time is unavailable.
7. **Thank the customer** and wish them a great day at the end.

## Important Rules
- ALWAYS confirm the full appointment details (service, stylist, date, time) with the customer before booking.
- If the caller is unclear, ask clarifying questions politely. Never guess.
- For rescheduling or cancellation, verify the existing appointment first by asking for their name or phone number.
- If a service or stylist isn't available, suggest alternatives.
- Keep your spoken responses SHORT and NATURAL – this is a phone call, not an email.
- Use 12-hour time format when speaking (e.g., "2:30 PM" not "14:30").
- When listing available times, offer the 2-3 best options rather than reading an entire list.
- Never share internal system details or error messages with the caller.
- If you can't help with something, politely suggest calling the salon directly or visiting the website.

## Date & Time Context
- Today's date will be provided to you. Use it to interpret relative dates like "tomorrow" or "next Tuesday".
- The salon's timezone is {settings.salon_timezone}.

## Example Interactions
- "Welcome to {salon_name}! How can I help you today?"
- "I'd love to help you book an appointment! What service are you looking for?"
- "Great choice! Let me check availability for you. Do you have a preferred stylist?"
- "I have openings at 10 AM and 2:30 PM on Thursday. Which works better for you?"
- "Perfect! I've booked your haircut with Sophia on Thursday at 2:30 PM. We'll see you then!"
- "I understand you'd like to reschedule. Can I have your name so I can pull up your appointment?"
"""


# ── Tool Definitions for OpenRouter / LLM ─────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check available appointment slots for a specific date, "
                "optionally filtered by stylist. Returns a list of open time windows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check in YYYY-MM-DD format.",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the service in minutes. Default 60.",
                        "default": 60,
                    },
                    "stylist": {
                        "type": "string",
                        "description": "Stylist name to filter availability by. Optional.",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a new appointment on Google Calendar. All fields are required. "
                "Only call this AFTER confirming details with the customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Name of the salon service.",
                    },
                    "stylist": {
                        "type": "string",
                        "description": "Name of the stylist.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Appointment date in YYYY-MM-DD format.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time in HH:MM (24-hour) format.",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration in minutes.",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's full name.",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Customer's phone number.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes. Optional.",
                    },
                },
                "required": [
                    "service",
                    "stylist",
                    "date",
                    "start_time",
                    "duration_minutes",
                    "customer_name",
                    "customer_phone",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": (
                "Reschedule an existing appointment to a new date/time. "
                "Requires the event_id from a prior lookup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "Google Calendar event ID of the appointment.",
                    },
                    "new_date": {
                        "type": "string",
                        "description": "New date in YYYY-MM-DD format.",
                    },
                    "new_start_time": {
                        "type": "string",
                        "description": "New start time in HH:MM (24-hour) format.",
                    },
                },
                "required": ["event_id", "new_date", "new_start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing appointment. Requires the event_id from a prior lookup. "
                "Remind the customer about the cancellation policy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "Google Calendar event ID of the appointment to cancel.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_appointment",
            "description": (
                "Find an existing appointment by customer name or phone number. "
                "Use this before rescheduling or cancelling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's name to search for.",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Customer's phone number to search for.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the salon knowledge base for information about services, "
                "pricing, stylists, policies, FAQs, location, and hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
