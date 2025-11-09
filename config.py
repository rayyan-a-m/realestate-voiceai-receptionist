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
# In production (like Render), set GOOGLE_CREDENTIALS_JSON as a secret
# environment variable with the content of your service account JSON file.
# For local development, you can set GOOGLE_APPLICATION_CREDENTIALS to the *path* of the JSON file.
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
            logging.warning("No explicit Google Cloud credentials provided (JSON or path). Falling back to ADC. This may not work in all environments.")
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
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash-lite")

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# --- Business & Calendar Configuration ---
YOUR_BUSINESS_NAME = "Prestige Properties"
APPOINTMENT_DURATION_MINUTES = 30
TIMEZONE = "America/New_York" # e.g., "America/New_York", "Europe/London"
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_JSON_PATH = "token.json" # For user OAuth flow

# --- Property Information (Static) ---
# This information will be used by the agent to answer questions.
PROPERTIES = [
    {
        "id": "prop123",
        "address": "123 Main St, Anytown, USA",
        "bedrooms": 3,
        "bathrooms": 2,
        "price": 500000,
        "features": "A beautiful family home with a large backyard and modern kitchen.",
    },
    {
        "id": "prop456",
        "address": "456 Oak Ave, Anytown, USA",
        "bedrooms": 4,
        "bathrooms": 3,
        "price": 750000,
        "features": "A spacious luxury home with a pool and a three-car garage.",
    },
]

