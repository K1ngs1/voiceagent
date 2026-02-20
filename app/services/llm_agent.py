"""
LLM Agent – OpenRouter Integration with Tool Calling.

Connects to OpenRouter (OpenAI-compatible API) for conversation
and dispatches tool calls to the appropriate services.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from openai import OpenAI

from app.config import get_settings
from app.prompts.salon_agent import get_system_prompt, TOOL_DEFINITIONS
from app.services.rag_service import rag_service
from app.services.calendar_service import calendar_service
from app.logger.interaction_logger import interaction_logger

logger = logging.getLogger(__name__)


class LLMAgent:
    """LLM-powered conversational agent via OpenRouter."""

    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._model: Optional[str] = None
        self._settings = None

    def initialize(self):
        """Set up the OpenAI client pointed at OpenRouter."""
        self._settings = get_settings()
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._settings.openrouter_api_key,
        )
        self._model = self._settings.openrouter_model
        logger.info(f"LLM Agent initialized with model: {self._model}")

    def _get_client(self) -> OpenAI:
        """Lazy-initialize and return the OpenAI client."""
        if self._client is None:
            self.initialize()
        return self._client

    def _build_system_message(self) -> dict:
        """Build the system message with current date context."""
        prompt = get_system_prompt()
        salon_tz = ZoneInfo(self._settings.salon_timezone if self._settings else "America/Los_Angeles")
        now = datetime.now(salon_tz)
        date_context = (
            f"\n\n## Current Date & Time\n"
            f"Today is {now.strftime('%A, %B %d, %Y')}. "
            f"Current time: {now.strftime('%I:%M %p')}."
        )
        return {"role": "system", "content": prompt + date_context}

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Execute a tool call and return the result as a string.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments parsed from the LLM's tool call.

        Returns:
            JSON string of the tool result.
        """
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            if tool_name == "check_availability":
                result = calendar_service.get_available_slots(
                    date=arguments["date"],
                    duration_minutes=arguments.get("duration_minutes", 60),
                    stylist=arguments.get("stylist"),
                )
                if not result:
                    return json.dumps({
                        "status": "no_slots",
                        "message": "No available slots found for the requested date and criteria.",
                    })
                # Limit to 5 slots for brevity
                return json.dumps({
                    "status": "available",
                    "slots": result[:5],
                    "total_available": len(result),
                })

            elif tool_name == "book_appointment":
                svc = rag_service.get_service_by_name(arguments["service"])
                duration = arguments.get(
                    "duration_minutes",
                    svc["duration_minutes"] if svc else 60,
                )
                summary = f"{arguments['service']} – {arguments['customer_name']}"
                result = calendar_service.create_appointment(
                    summary=summary,
                    date=arguments["date"],
                    start_time=arguments["start_time"],
                    duration_minutes=duration,
                    customer_name=arguments["customer_name"],
                    customer_phone=arguments["customer_phone"],
                    service=arguments["service"],
                    stylist=arguments.get("stylist", ""),
                    notes=arguments.get("notes", ""),
                )
                return json.dumps(result)

            elif tool_name == "reschedule_appointment":
                result = calendar_service.update_appointment(
                    event_id=arguments["event_id"],
                    new_date=arguments["new_date"],
                    new_start_time=arguments["new_start_time"],
                )
                return json.dumps(result)

            elif tool_name == "cancel_appointment":
                result = calendar_service.delete_appointment(
                    event_id=arguments["event_id"],
                )
                return json.dumps(result)

            elif tool_name == "lookup_appointment":
                result = calendar_service.find_appointment(
                    customer_name=arguments.get("customer_name"),
                    customer_phone=arguments.get("customer_phone"),
                )
                if not result:
                    return json.dumps({
                        "status": "not_found",
                        "message": "No upcoming appointment found with that information.",
                    })
                return json.dumps({"status": "found", "appointments": result})

            elif tool_name == "search_knowledge_base":
                results = rag_service.query(
                    question=arguments["query"],
                    top_k=3,
                )
                return json.dumps({
                    "status": "success",
                    "results": results,
                })

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return json.dumps({"error": str(e)})

    def chat(
        self,
        conversation_history: list[dict],
        user_message: str,
        call_sid: str = "unknown",
    ) -> tuple[str, list[dict], list[str]]:
        """
        Send a user message to the LLM and get a response,
        executing any tool calls along the way.

        Args:
            conversation_history: Existing messages in the conversation.
            user_message: The latest user (customer) message from STT.
            call_sid: Twilio Call SID for logging.

        Returns:
            Tuple of (agent_response_text, updated_conversation_history, tools_called_list).
        """
        client = self._get_client()

        # Trim history if too long to prevent context window overflow
        MAX_HISTORY = 40
        if len(conversation_history) > MAX_HISTORY:
            # Keep first 4 messages (opening context) and last 36 (recent)
            conversation_history = conversation_history[:4] + conversation_history[-36:]

        # Build messages
        messages = [self._build_system_message()]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        tools_called = []
        max_tool_rounds = 5  # Safety limit to prevent infinite tool loops

        for round_num in range(max_tool_rounds):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=500,
                )
            except Exception as e:
                logger.error(f"OpenRouter API error: {e}")
                interaction_logger.log_error(call_sid, str(e), "llm_chat")
                return (
                    "I'm sorry, I'm having a bit of trouble right now. "
                    "Could you repeat that?",
                    conversation_history,
                    tools_called,
                )

            choice = response.choices[0]

            # If the LLM wants to call tools
            if choice.message.tool_calls:
                # Add the assistant message with tool calls
                messages.append(choice.message.model_dump())

                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    tools_called.append(fn_name)
                    tool_result = self._execute_tool(fn_name, fn_args)

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )

                # Continue the loop to let the LLM process tool results
                continue

            # No more tool calls – we have the final response
            agent_response = choice.message.content or ""

            # Update conversation history (exclude system message)
            updated_history = conversation_history.copy()
            updated_history.append({"role": "user", "content": user_message})
            updated_history.append({"role": "assistant", "content": agent_response})

            # Log the interaction
            interaction_logger.log_interaction(
                call_sid=call_sid,
                customer_transcript=user_message,
                agent_response=agent_response,
                tools_called=tools_called,
            )

            return agent_response, updated_history, tools_called

        # If we exhausted tool rounds
        logger.warning(f"Tool execution loop exceeded {max_tool_rounds} rounds")
        return (
            "I apologize for the delay. Let me help you with that. "
            "Could you tell me what you'd like to do?",
            conversation_history,
            tools_called,
        )

    def get_greeting(self) -> str:
        """Generate the initial greeting for an incoming call."""
        settings = get_settings()
        return (
            f"Welcome to {settings.salon_name}! "
            f"Thank you for calling. How can I help you today?"
        )


# Singleton instance
llm_agent = LLMAgent()
