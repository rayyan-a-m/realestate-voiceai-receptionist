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
from config import APPOINTMENT_DURATION_MINUTES, TIMEZONE, PROPERTIES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Calendar API Setup ---
SCOPES = ["https://www.googleapis.com/auth/calendar"]
# The ID of the secret in Google Secret Manager containing the OAuth token JSON.
OAUTH_TOKEN_SECRET_ID = os.getenv("OAUTH_TOKEN_SECRET_ID")


def get_property_by_id(property_id: str):
    """Helper function to find a property by its ID."""
    for prop in PROPERTIES:
        if prop['id'] == property_id:
            return prop
    return None


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
def find_available_slots(date_str: str) -> str:
    """
    Use this tool to find available appointment slots on a specific day.

    You MUST ask the user for a specific date before using this tool.

    Args:
        date_str: The date to check for availability in 'YYYY-MM-DD' format.

    This tool is the ONLY way to check for appointment availability.
    Do not guess or suggest times without calling this tool first.
    It returns a string with available 30-minute slots for the given day.
    """
    logging.info(f"Tool 'find_available_slots' invoked for date: {date_str}")
    try:
        service = get_calendar_service()

        # Parse the input date and set the time range for that day
        try:
            # Ensure date_str is just the date part if datetime is passed
            if ' ' in date_str:
                date_str = date_str.split(' ')[0]
            search_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            logging.error(f"Invalid date format received: '{date_str}'")
            return "Error: Invalid date format. Please ask the user for a date in 'YYYY-MM-DD' format."

        tz = pytz.timezone(TIMEZONE)
        # Define working hours (e.g., 9 AM to 5 PM)
        day_start = tz.localize(datetime.datetime.combine(search_date, datetime.time(9, 0)))
        day_end = tz.localize(datetime.datetime.combine(search_date, datetime.time(17, 0)))
        
        # Ensure we don't search in the past
        now = datetime.datetime.now(tz)
        if day_start < now:
            day_start = now

        logging.info(f"Searching for free slots on {date_str} between {day_start.isoformat()} and {day_end.isoformat()}.")

        events_result = service.events().list(
            calendarId='primary',
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        busy_slots = events_result.get('items', [])
        logging.info(f"Found {len(busy_slots)} busy slots in the calendar for {date_str}.")

        available_slots = []
        # Start checking from the beginning of the workday or now, whichever is later
        potential_slot_time = day_start
        # Align to the next 30-minute mark
        if potential_slot_time.minute not in [0, 30]:
             if potential_slot_time.minute > 30:
                 potential_slot_time = potential_slot_time.replace(minute=30) + datetime.timedelta(minutes=30)
             else:
                 potential_slot_time = potential_slot_time.replace(minute=30)
        
        while potential_slot_time < day_end:
            is_free = True
            slot_end_time = potential_slot_time + datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
            
            for event in busy_slots:
                event_start = datetime.datetime.fromisoformat(event['start'].get('dateTime')).astimezone(tz)
                event_end = datetime.datetime.fromisoformat(event['end'].get('dateTime')).astimezone(tz)
                
                # Check for overlap
                if max(potential_slot_time, event_start) < min(slot_end_time, event_end):
                    is_free = False
                    logging.debug(f"Slot at {potential_slot_time} conflicts with event '{event.get('summary')}' from {event_start} to {event_end}.")
                    # Move potential start time to the end of the conflicting event
                    potential_slot_time = event_end
                    # Re-align to 30-min interval
                    if potential_slot_time.minute not in [0, 30]:
                        if potential_slot_time.minute > 30:
                            potential_slot_time = potential_slot_time.replace(hour=potential_slot_time.hour + 1, minute=0)
                        else:
                            potential_slot_time = potential_slot_time.replace(minute=30)
                    break 
            
            if is_free:
                # Add slot if it's within the working day
                if potential_slot_time < day_end:
                    available_slots.append(potential_slot_time.strftime('%H:%M'))
                potential_slot_time += datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)

        if not available_slots:
            logging.warning(f"No available slots found for {date_str}.")
            return f"I'm sorry, but there are no available slots on {date_str}. Would you like to check another date?"

        result_str = f"On {date_str}, the following times are available: {', '.join(available_slots)}."
        logging.info(f"Found available slots for {date_str}: {result_str}")
        return result_str
    except Exception as e:
        logging.error(f"Error in find_available_slots: {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to find available slots. Could you please try another date?"

@tool
def book_appointment(datetime_str: str, full_name: str, email: str, property_id: str) -> str:
    """
    Use this tool to book a property visit appointment.

    This is the final step in the booking process.
    You MUST have the user's full name, email, the desired datetime_str,
    and the property_id BEFORE calling this tool.

    Args:
        datetime_str: The appointment time in 'YYYY-MM-DD HH:MM' format (24-hour clock),
                      which MUST be one of the slots provided by 'find_available_slots'.
        full_name: The full name of the person booking the visit.
        email: The email address of the person.
        property_id: The ID of the property they want to visit (e.g., 'PV001').
    """
    logging.info(f"Tool 'book_appointment' invoked with: datetime='{datetime_str}', name='{full_name}', email='{email}', property_id='{property_id}'")

    property_details = get_property_by_id(property_id)
    if not property_details:
        logging.error(f"Invalid property_id '{property_id}' passed to book_appointment.")
        return f"Error: I couldn't find a property with the ID '{property_id}'. Please confirm the property ID."

    property_name = property_details.get('address', 'Unknown Property')

    try:
        service = get_calendar_service()
        start_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M").astimezone(pytz.timezone(TIMEZONE))
        end_time = start_time + datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
    except ValueError:
        logging.error(f"Invalid datetime format received: '{datetime_str}'")
        return "Error: Invalid datetime format. Please use 'YYYY-MM-DD HH:MM'."

    event = {
        'summary': f'Property Visit: {property_name} for {full_name}',
        'description': f'Booked by AI Assistant "Sky".\nClient Email: {email}\nProperty ID: {property_id}',
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
        return f"Success! The appointment for {full_name} at {datetime_str} for the property at {property_name} has been booked. A calendar invite has been sent."
    except Exception as e:
        logging.error(f"Failed to create calendar event: {e}", exc_info=True)
        return "Sorry, I was unable to book the appointment. There was an error with the calendar service."

# This allows the script to be run directly to generate the initial token.json
if __name__ == '__main__':
    logging.info("Running google_calendar.py directly to authenticate and generate token.json...")
    get_calendar_service()
    logging.info("Authentication successful. token.json should now be present.")
