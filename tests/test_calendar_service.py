"""
Tests for the Calendar Service.

Uses mocks – no real Google Calendar API calls are made.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.calendar_service import CalendarService


@pytest.fixture
def mock_settings():
    """Mock settings for the calendar service."""
    settings = MagicMock()
    settings.google_calendar_id = "test-calendar@group.calendar.google.com"
    settings.google_service_account_file = "test-creds.json"
    settings.salon_timezone = "America/Los_Angeles"
    return settings


@pytest.fixture
def calendar_svc(mock_settings):
    """Create a CalendarService with mocked Google API."""
    svc = CalendarService()
    svc._settings = mock_settings

    # Mock the Google Calendar API service
    mock_service = MagicMock()
    svc._service = mock_service
    return svc


class TestAvailability:
    """Test availability checking logic."""

    def test_get_slots_empty_calendar(self, calendar_svc):
        """Should return all slots when calendar is empty."""
        # Mock: no events on the calendar
        mock_events = {"items": []}
        calendar_svc._service.events().list().execute.return_value = mock_events

        slots = calendar_svc.get_available_slots(
            date="2026-03-15", duration_minutes=60
        )
        assert len(slots) > 0
        assert all("start_time" in s and "end_time" in s for s in slots)

    def test_get_slots_with_busy_time(self, calendar_svc):
        """Should exclude busy slots from available times."""
        mock_events = {
            "items": [
                {
                    "start": {"dateTime": "2026-03-15T10:00:00-07:00"},
                    "end": {"dateTime": "2026-03-15T11:00:00-07:00"},
                    "summary": "Existing appointment",
                }
            ]
        }
        calendar_svc._service.events().list().execute.return_value = mock_events

        slots = calendar_svc.get_available_slots(
            date="2026-03-15", duration_minutes=60
        )
        # 10:00 AM slot should not be available
        start_times = [s["start_time"] for s in slots]
        assert "10:00 AM" not in start_times

    def test_get_slots_error_handling(self, calendar_svc):
        """Should return empty list on API error."""
        calendar_svc._service.events().list().execute.side_effect = Exception("API error")

        slots = calendar_svc.get_available_slots(date="2026-03-15")
        assert slots == []


class TestAppointmentCreation:
    """Test appointment creation."""

    def test_create_appointment(self, calendar_svc):
        """Should create a calendar event and return confirmation."""
        mock_response = {
            "id": "event123",
            "summary": "Haircut – Jane Doe",
            "htmlLink": "https://calendar.google.com/event123",
        }
        calendar_svc._service.events().insert().execute.return_value = mock_response

        result = calendar_svc.create_appointment(
            summary="Haircut – Jane Doe",
            date="2026-03-15",
            start_time="14:00",
            duration_minutes=60,
            customer_name="Jane Doe",
            customer_phone="+15551234567",
            service="Women's Haircut & Style",
            stylist="Sophia Martinez",
        )

        assert result["event_id"] == "event123"
        assert result["status"] == "confirmed"
        assert result["customer_name"] == "Jane Doe"

    def test_create_appointment_error(self, calendar_svc):
        """Should raise on creation error."""
        calendar_svc._service.events().insert().execute.side_effect = Exception(
            "Create failed"
        )

        with pytest.raises(Exception, match="Create failed"):
            calendar_svc.create_appointment(
                summary="Test",
                date="2026-03-15",
                start_time="10:00",
                duration_minutes=60,
                customer_name="Test",
                customer_phone="+1555",
                service="Test",
            )


class TestAppointmentLookup:
    """Test finding existing appointments."""

    def test_find_by_name(self, calendar_svc):
        """Should find appointments matching customer name."""
        mock_events = {
            "items": [
                {
                    "id": "event456",
                    "summary": "Haircut – Jane Doe",
                    "description": "Customer: Jane Doe\nPhone: +15551234567",
                    "start": {"dateTime": "2026-03-20T14:00:00Z"},
                    "end": {"dateTime": "2026-03-20T15:00:00Z"},
                }
            ]
        }
        calendar_svc._service.events().list().execute.return_value = mock_events

        results = calendar_svc.find_appointment(customer_name="jane doe")
        assert len(results) == 1
        assert results[0]["event_id"] == "event456"

    def test_find_not_found(self, calendar_svc):
        """Should return empty list when no match found."""
        mock_events = {"items": []}
        calendar_svc._service.events().list().execute.return_value = mock_events

        results = calendar_svc.find_appointment(customer_name="Nobody")
        assert results == []


class TestAppointmentModification:
    """Test rescheduling and cancellation."""

    def test_update_appointment(self, calendar_svc):
        """Should update the event time."""
        existing_event = {
            "id": "event789",
            "summary": "Haircut – Jane",
            "start": {"dateTime": "2026-03-15T14:00:00Z"},
            "end": {"dateTime": "2026-03-15T15:00:00Z"},
        }
        calendar_svc._service.events().get().execute.return_value = existing_event

        updated_event = {
            "id": "event789",
            "summary": "Haircut – Jane",
            "start": {"dateTime": "2026-03-16T10:00:00"},
            "end": {"dateTime": "2026-03-16T11:00:00"},
        }
        calendar_svc._service.events().update().execute.return_value = updated_event

        result = calendar_svc.update_appointment(
            event_id="event789",
            new_date="2026-03-16",
            new_start_time="10:00",
        )
        assert result["status"] == "rescheduled"

    def test_delete_appointment(self, calendar_svc):
        """Should delete the event."""
        existing_event = {
            "id": "event999",
            "summary": "Blowout – Sarah",
        }
        calendar_svc._service.events().get().execute.return_value = existing_event
        calendar_svc._service.events().delete().execute.return_value = None

        result = calendar_svc.delete_appointment(event_id="event999")
        assert result["status"] == "cancelled"
        assert result["event_id"] == "event999"
