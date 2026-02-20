"""
Google Calendar API Service.

Handles checking availability, creating, updating, and deleting
appointments using a Google Service Account.
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config import get_settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarService:
    """Google Calendar integration for appointment management."""

    def __init__(self):
        self._service = None
        self._settings = None

    def initialize(self):
        """Authenticate and build the Google Calendar service client."""
        self._settings = get_settings()
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self._settings.google_service_account_file, scopes=SCOPES
            )
            self._service = build("calendar", "v3", credentials=credentials)
            logger.info("Google Calendar service initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar: {e}")
            raise

    def _get_service(self):
        """Lazy-initialize and return the Calendar API service."""
        if self._service is None:
            self.initialize()
        return self._service

    def _calendar_id(self) -> str:
        """Return the configured calendar ID."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings.google_calendar_id

    def _timezone(self) -> str:
        """Return the configured salon timezone."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings.salon_timezone

    def get_available_slots(
        self,
        date: str,
        duration_minutes: int = 60,
        stylist: Optional[str] = None,
    ) -> list[dict]:
        """
        Check Google Calendar for available time slots on a given date.

        Args:
            date: Date string in YYYY-MM-DD format.
            duration_minutes: Service duration in minutes.
            stylist: Optional stylist name to filter by.

        Returns:
            List of dicts with 'start_time' and 'end_time' strings.
        """
        service = self._get_service()
        tz = self._timezone()

        # Define business hours (9 AM to 7 PM) in the salon's timezone
        tz_info = ZoneInfo(tz)
        day_start = datetime.fromisoformat(f"{date}T09:00:00").replace(tzinfo=tz_info)
        day_end = datetime.fromisoformat(f"{date}T19:00:00").replace(tzinfo=tz_info)

        # Query existing events for the day
        time_min = day_start.isoformat()
        time_max = day_end.isoformat()

        try:
            events_result = (
                service.events()
                .list(
                    calendarId=self._calendar_id(),
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])
        except Exception as e:
            logger.error(f"Error fetching calendar events: {e}")
            return []

        # Filter by stylist if specified
        if stylist:
            events = [
                e
                for e in events
                if stylist.lower() in e.get("description", "").lower()
                or stylist.lower() in e.get("summary", "").lower()
            ]

        # Build list of busy periods, normalized to salon timezone
        busy_periods = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            end = event["end"].get("dateTime", event["end"].get("date"))
            bs = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(tz_info)
            be = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(tz_info)
            busy_periods.append((bs, be))

        # Find available slots
        available_slots = []
        current = day_start
        duration = timedelta(minutes=duration_minutes)

        while current + duration <= day_end:
            slot_end = current + duration
            # Check if this slot conflicts with any busy period
            is_free = True
            for busy_start, busy_end in busy_periods:
                if current < busy_end and slot_end > busy_start:
                    is_free = False
                    # Jump to end of this busy period
                    current = busy_end
                    break

            if is_free:
                available_slots.append(
                    {
                        "start_time": current.strftime("%H:%M"),
                        "end_time": slot_end.strftime("%H:%M"),
                        "date": date,
                    }
                )
                current += timedelta(minutes=30)  # 30-min intervals
            # If not free, current was already advanced above

        return available_slots

    def create_appointment(
        self,
        summary: str,
        date: str,
        start_time: str,
        duration_minutes: int,
        customer_name: str,
        customer_phone: str,
        service: str,
        stylist: str = "",
        notes: str = "",
    ) -> dict:
        """
        Create a new calendar event for an appointment.

        Args:
            summary: Event title (e.g. "Haircut – Jane Doe").
            date: Date string YYYY-MM-DD.
            start_time: Start time string HH:MM (24h).
            duration_minutes: Duration in minutes.
            customer_name: Customer's name.
            customer_phone: Customer's phone.
            service: Service booked.
            stylist: Stylist name.
            notes: Extra notes.

        Returns:
            Dict with event details including 'event_id'.
        """
        service_api = self._get_service()
        tz = self._timezone()

        start_dt = datetime.fromisoformat(f"{date}T{start_time}:00")
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        description = (
            f"Customer: {customer_name}\n"
            f"Phone: {customer_phone}\n"
            f"Service: {service}\n"
            f"Stylist: {stylist}\n"
            f"Notes: {notes}"
        )

        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": tz,
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": tz,
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 60},
                ],
            },
        }

        try:
            created_event = (
                service_api.events()
                .insert(calendarId=self._calendar_id(), body=event)
                .execute()
            )
            logger.info(f"Appointment created: {created_event.get('id')}")
            return {
                "event_id": created_event["id"],
                "summary": summary,
                "date": date,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "customer_name": customer_name,
                "service": service,
                "stylist": stylist,
                "status": "confirmed",
                "link": created_event.get("htmlLink", ""),
            }
        except Exception as e:
            logger.error(f"Error creating appointment: {e}")
            raise

    def find_appointment(
        self,
        customer_name: Optional[str] = None,
        customer_phone: Optional[str] = None,
        date_from: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for existing appointments by customer name or phone.

        Args:
            customer_name: Customer name to search for.
            customer_phone: Customer phone to search for.
            date_from: Only search from this date forward (YYYY-MM-DD). Defaults to today.

        Returns:
            List of matching event dicts.
        """
        service = self._get_service()

        if date_from is None:
            date_from = datetime.now().strftime("%Y-%m-%d")

        time_min = f"{date_from}T00:00:00Z"
        # Search up to 90 days ahead
        time_max = (
            datetime.fromisoformat(date_from) + timedelta(days=90)
        ).isoformat() + "Z"

        try:
            events_result = (
                service.events()
                .list(
                    calendarId=self._calendar_id(),
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=50,
                )
                .execute()
            )
            events = events_result.get("items", [])
        except Exception as e:
            logger.error(f"Error searching appointments: {e}")
            return []

        matches = []
        search_term = (customer_name or customer_phone or "").lower()

        for event in events:
            desc = event.get("description", "").lower()
            summary = event.get("summary", "").lower()

            if search_term and (search_term in desc or search_term in summary):
                start = event["start"].get("dateTime", event["start"].get("date"))
                end = event["end"].get("dateTime", event["end"].get("date"))
                matches.append(
                    {
                        "event_id": event["id"],
                        "summary": event.get("summary", ""),
                        "description": event.get("description", ""),
                        "start": start,
                        "end": end,
                    }
                )

        return matches

    def update_appointment(
        self,
        event_id: str,
        new_date: Optional[str] = None,
        new_start_time: Optional[str] = None,
        new_duration_minutes: Optional[int] = None,
    ) -> dict:
        """
        Reschedule an existing appointment.

        Args:
            event_id: Google Calendar event ID.
            new_date: New date (YYYY-MM-DD).
            new_start_time: New start time (HH:MM, 24h).
            new_duration_minutes: New duration if changed.

        Returns:
            Updated event dict.
        """
        service = self._get_service()
        tz = self._timezone()

        try:
            # Fetch existing event
            event = (
                service.events()
                .get(calendarId=self._calendar_id(), eventId=event_id)
                .execute()
            )

            if new_date and new_start_time:
                start_dt = datetime.fromisoformat(f"{new_date}T{new_start_time}:00")
                # Calculate duration from old event if not provided
                if new_duration_minutes is None:
                    old_start = datetime.fromisoformat(
                        event["start"]["dateTime"].replace("Z", "+00:00")
                    )
                    old_end = datetime.fromisoformat(
                        event["end"]["dateTime"].replace("Z", "+00:00")
                    )
                    new_duration_minutes = int(
                        (old_end - old_start).total_seconds() / 60
                    )

                end_dt = start_dt + timedelta(minutes=new_duration_minutes)

                event["start"] = {"dateTime": start_dt.isoformat(), "timeZone": tz}
                event["end"] = {"dateTime": end_dt.isoformat(), "timeZone": tz}

            updated_event = (
                service.events()
                .update(
                    calendarId=self._calendar_id(),
                    eventId=event_id,
                    body=event,
                )
                .execute()
            )

            logger.info(f"Appointment updated: {event_id}")
            return {
                "event_id": event_id,
                "summary": updated_event.get("summary", ""),
                "new_start": updated_event["start"].get("dateTime", ""),
                "new_end": updated_event["end"].get("dateTime", ""),
                "status": "rescheduled",
            }
        except Exception as e:
            logger.error(f"Error updating appointment: {e}")
            raise

    def delete_appointment(self, event_id: str) -> dict:
        """
        Cancel (delete) an appointment.

        Args:
            event_id: Google Calendar event ID.

        Returns:
            Confirmation dict.
        """
        service = self._get_service()

        try:
            # First fetch the event to confirm details
            event = (
                service.events()
                .get(calendarId=self._calendar_id(), eventId=event_id)
                .execute()
            )
            summary = event.get("summary", "Unknown")

            # Delete the event
            service.events().delete(
                calendarId=self._calendar_id(), eventId=event_id
            ).execute()

            logger.info(f"Appointment cancelled: {event_id} ({summary})")
            return {
                "event_id": event_id,
                "summary": summary,
                "status": "cancelled",
            }
        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}")
            raise


# Singleton instance
calendar_service = CalendarService()
