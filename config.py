import os
import json
import logging
from dotenv import load_dotenv
from google.oauth2 import service_account

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Cloud Credentials ---
# This is the central source of truth for all Google Cloud authentication.
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
GOOGLE_APPLICATION_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def get_google_credentials():
    """Loads Google Cloud credentials from env vars or a file."""
    creds = None
    try:
        if GOOGLE_CREDENTIALS_JSON:
            credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)
            creds = service_account.Credentials.from_service_account_info(credentials_info)
            logging.info("Loaded Google Cloud credentials from GOOGLE_CREDENTIALS_JSON.")
        elif GOOGLE_APPLICATION_CREDENTIALS_PATH and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS_PATH):
            creds = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS_PATH)
            logging.info(f"Loaded Google Cloud credentials from file: {GOOGLE_APPLICATION_CREDENTIALS_PATH}")
        else:
            # This is not an error, but a fallback to Application Default Credentials (ADC)
            logging.warning("No explicit Google Cloud credentials provided (JSON or path). Falling back to ADC.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse GOOGLE_CREDENTIALS_JSON: {e}")
    except Exception as e:
        logging.error(f"Failed to load Google Cloud credentials: {e}")
    return creds

# Centralized credentials object to be used across the application
GOOGLE_CREDENTIALS = get_google_credentials()

# --- Project Configuration ---
# Attempt to get project_id from credentials if available
project_id_from_creds = GOOGLE_CREDENTIALS.project_id if GOOGLE_CREDENTIALS else None
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") or project_id_from_creds or os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-1.5-flash-001")

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# --- Calendar Configuration ---
APPOINTMENT_DURATION_MINUTES = 30
TIMEZONE = "America/Los_Angeles"
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_JSON_PATH = "token.json" # For user OAuth flow

# --- Property Information (Static) ---
PROPERTY_INFO = {
    "address": "123 Main Street, Anytown, USA",
    "bedrooms": 3,
    "bathrooms": 2,
    "price": 500000,
    "features": "a large backyard, a newly renovated kitchen, and a two-car garage.",
}
