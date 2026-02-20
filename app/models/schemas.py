"""
Pydantic schemas for the Salon AI Voice Agent.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AppointmentRequest(BaseModel):
    """Request to book an appointment."""

    service: str = Field(..., description="Salon service requested")
    stylist: Optional[str] = Field(None, description="Preferred stylist")
    date: str = Field(..., description="Requested date (YYYY-MM-DD)")
    time: str = Field(..., description="Requested time (HH:MM)")
    customer_name: str = Field(..., description="Customer's full name")
    customer_phone: str = Field(..., description="Customer's phone number")
    notes: Optional[str] = Field(None, description="Additional notes")


class AppointmentResponse(BaseModel):
    """Response after booking an appointment."""

    event_id: str = Field(..., description="Google Calendar event ID")
    service: str
    stylist: str
    confirmed_date: str
    confirmed_time: str
    duration_minutes: int
    customer_name: str
    status: str = Field(default="confirmed")


class AvailableSlot(BaseModel):
    """An available time slot."""

    date: str
    start_time: str
    end_time: str
    stylist: str


class RAGQuery(BaseModel):
    """Query to the knowledge base."""

    question: str = Field(..., description="Question to search the knowledge base for")
    top_k: int = Field(default=3, description="Number of results to retrieve")


class RAGResult(BaseModel):
    """Result from the knowledge base."""

    content: str = Field(..., description="Retrieved content")
    source: str = Field(..., description="Source section of the knowledge base")
    relevance_score: float = Field(..., description="Similarity score")


class CallSession(BaseModel):
    """Tracks state for an active phone call."""

    call_sid: str = Field(..., description="Twilio Call SID")
    customer_phone: Optional[str] = Field(None, description="Caller phone number")
    customer_name: Optional[str] = Field(None, description="Extracted customer name")
    conversation_history: list[dict] = Field(
        default_factory=list, description="List of message dicts for the LLM"
    )
    extracted_info: dict = Field(
        default_factory=dict, description="Info extracted so far from conversation"
    )
    intent: Optional[str] = Field(
        None, description="Detected intent: book, reschedule, cancel, inquiry"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="Call start time"
    )
    is_active: bool = Field(default=True, description="Whether the call is still active")


class InteractionLog(BaseModel):
    """A single logged interaction."""

    timestamp: str
    call_sid: str
    direction: str = "inbound"
    customer_phone: Optional[str] = None
    customer_transcript: Optional[str] = None
    agent_response: Optional[str] = None
    intent_detected: Optional[str] = None
    tools_called: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
