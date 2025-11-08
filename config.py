import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
AGENT_HOST_URL = os.getenv("AGENT_HOST_URL")
YOUR_BUSINESS_NAME = os.getenv("YOUR_BUSINESS_NAME")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
TIMEZONE = "America/Los_Angeles"
APPOINTMENT_DURATION_MINUTES = 30
PROPERTIES = [
    {
        "name": "Sunset Villas",
        "type": "Luxury Villa",
        "location": "Malibu, California",
        "description": "Stunning oceanfront villas with modern amenities and breathtaking views. Perfect for a serene and upscale lifestyle.",
        "talking_point": "the incredible sunset views and private beach access"
    },
    {
        "name": "Metropolitan Lofts",
        "type": "Urban Loft",
        "location": "Downtown, Los Angeles",
        "description": "Chic and spacious lofts in the heart of the city. Features high ceilings, large windows, and a vibrant, artistic community.",
        "talking_point": "the immediate access to downtown's best restaurants and galleries"
    }
]

# --- Service Account Credentials ---
# Provides access to Google Cloud services
try:
    from google.oauth2 import service_account
    # Load credentials from the specified path
    CREDENTIALS = service_account.Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
except Exception as e:
    CREDENTIALS = None
    print(f"Credential loading failed: {e}")
