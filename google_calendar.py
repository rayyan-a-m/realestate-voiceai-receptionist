import os
import datetime
import pytz
import logging
import json
from langchain_core.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.cloud import secretmanager

import config
from config import APPOINTMENT_DURATION_MINUTES, TIMEZONE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Calendar API Setup ---
SCOPES = ["https://www.googleapis.com/auth/calendar"]
# The ID of the secret in Google Secret Manager containing the OAuth token JSON.
OAUTH_TOKEN_SECRET_ID = os.getenv("OAUTH_TOKEN_SECRET_ID")


def get_calendar_service():
    """
    Gets an authorized Google Calendar service instance.

    Priority order for credentials:
    1. Fetches from Google Secret Manager if OAUTH_TOKEN_SECRET_ID is set.
    2. Reads from a local 'token.json' file.
    3. Initiates a new OAuth 2.0 flow as a last resort.
    """
    creds = None

    # 1. Try to load from Google Secret Manager
    if OAUTH_TOKEN_SECRET_ID and config.GCP_PROJECT_ID:
        try:
            logging.info(f"Attempting to load OAuth token from Secret Manager: '{OAUTH_TOKEN_SECRET_ID}'")
            client = secretmanager.SecretManagerServiceClient(credentials=config.GOOGLE_CREDENTIALS)
            secret_name = f"projects/{config.GCP_PROJECT_ID}/secrets/{OAUTH_TOKEN_SECRET_ID}/versions/latest"
            response = client.access_secret_version(request={"name": secret_name})
            token_data = json.loads(response.payload.data.decode("UTF-8"))
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
            logging.info("Successfully loaded credentials from Secret Manager.")
        except Exception as e:
            logging.error(f"Failed to load token from Secret Manager. Falling back. Error: {e}", exc_info=True)

    # 2. If Secret Manager fails or is not configured, try local token.json
    if not creds and os.path.exists("token.json"):
        logging.info("Found token.json, loading credentials from file.")
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # 3. Refresh or run new OAuth flow if credentials are not valid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info("Credentials expired, refreshing token.")
            creds.refresh(Request())
        else:
            logging.info("No valid credentials found, running new OAuth flow.")
            # This part is for local development and should not run in production
            if not os.path.exists("credentials.json"):
                logging.error("FATAL: credentials.json not found. Cannot initiate OAuth flow.")
                raise FileNotFoundError("credentials.json is required to create a new token.")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the new/refreshed credentials to token.json for local cache
        try:
            with open("token.json", "w") as token:
                logging.info("Saving/updating credentials to token.json.")
                token.write(creds.to_json())
        except IOError as e:
            logging.error(f"Could not write to token.json: {e}")
            
    return build("calendar", "v3", credentials=creds)

@tool
def find_available_slots() -> str:
    """
    Finds the next 5 available 30-minute appointment slots starting from now.
    This tool does not require any input. It checks the calendar 24/7.
    """
    logging.info("Tool 'find_available_slots' invoked.")
    try:
        service = get_calendar_service()
        now = datetime.datetime.now(pytz.timezone(TIMEZONE))
        
        # Search for the next 7 days
        search_end = now + datetime.timedelta(days=7)
        logging.info(f"Searching for free slots between {now.isoformat()} and {search_end.isoformat()}.")

        events_result = service.events().list(
            calendarId='primary',
            timeMin=now.isoformat(),
            timeMax=search_end.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        busy_slots = events_result.get('items', [])
        logging.info(f"Found {len(busy_slots)} busy slots in the calendar.")

        available_slots = []
        # Start checking from the next whole 30-minute interval
        potential_slot_time = now.replace(second=0, microsecond=0)
        if potential_slot_time.minute >= 30:
            potential_slot_time = potential_slot_time.replace(minute=30)
        else:
            potential_slot_time = potential_slot_time.replace(minute=0)
        potential_slot_time += datetime.timedelta(minutes=30)
        
        limit = 5 # Find up to 5 slots
        while potential_slot_time < search_end and len(available_slots) < limit:
            is_free = True
            slot_end_time = potential_slot_time + datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
            
            for event in busy_slots:
                event_start = datetime.datetime.fromisoformat(event['start'].get('dateTime')).astimezone(pytz.timezone(TIMEZONE))
                event_end = datetime.datetime.fromisoformat(event['end'].get('dateTime')).astimezone(pytz.timezone(TIMEZONE))
                
                # Check for overlap
                if max(potential_slot_time, event_start) < min(slot_end_time, event_end):
                    is_free = False
                    # Use debug logging for verbose info
                    logging.debug(f"Slot at {potential_slot_time} conflicts with event '{event.get('summary')}' from {event_start} to {event_end}.")
                    break
            
            if is_free:
                available_slots.append(potential_slot_time.strftime('%Y-%m-%d %H:%M'))

            potential_slot_time += datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)

        if not available_slots:
            logging.warning("No available slots found in the next 7 days.")
            return "No available slots found in the next 7 days."

        result_str = f"Here are the next available slots: {', '.join(available_slots)}"
        logging.info(f"Found available slots: {result_str}")
        return result_str
    except Exception as e:
        logging.error(f"Error in find_available_slots: {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to find available slots."

@tool
def book_appointment(datetime_str: str, full_name: str, email: str, property_name: str) -> str:
    """Books a 30-minute property visit appointment.

    Args:
        datetime_str: The appointment time in 'YYYY-MM-DD HH:MM' format (24-hour clock).
        full_name: The full name of the person booking the visit.
        email: The email address of the person.
        property_name: The name of the property they want to visit.
    """
    logging.info(f"Tool 'book_appointment' invoked with: datetime='{datetime_str}', name='{full_name}', email='{email}', property='{property_name}'")
    try:
        service = get_calendar_service()
        start_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M").astimezone(pytz.timezone(TIMEZONE))
        end_time = start_time + datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
    except ValueError:
        logging.error(f"Invalid datetime format received: '{datetime_str}'")
        return "Error: Invalid datetime format. Please use 'YYYY-MM-DD HH:MM'."

    event = {
        'summary': f'Property Visit: {property_name} for {full_name}',
        'description': f'Booked by AI Assistant "Sky".\nClient Email: {email}',
        'start': {'dateTime': start_time.isoformat(), 'timeZone': TIMEZONE},
        'end': {'dateTime': end_time.isoformat(), 'timeZone': TIMEZONE},
        'attendees': [{'email': email}],
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 60},
            ],
        },
    }

    try:
        created_event = service.events().insert(calendarId='primary', body=event, sendUpdates='all').execute()
        logging.info(f"Successfully created event with ID: {created_event.get('id')}")
        return f"Success! The appointment for {full_name} at {datetime_str} has been booked. A calendar invite has been sent."
    except Exception as e:
        logging.error(f"Failed to create calendar event: {e}", exc_info=True)
        return "Sorry, I was unable to book the appointment. There was an error with the calendar service."

# This allows the script to be run directly to generate the initial token.json
if __name__ == '__main__':
    logging.info("Running google_calendar.py directly to authenticate and generate token.json...")
    get_calendar_service()
    logging.info("Authentication successful. token.json should now be present.")
