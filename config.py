import os
from dotenv import load_dotenv

load_dotenv()

# --- Voice Agent Configuration ---
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Example: "Rachel"

# --- API Keys and Credentials ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# --- Application Configuration ---
AGENT_HOST_URL = os.getenv("AGENT_HOST_URL")

# --- Business/Personal Configuration ---
# UPDATED for Real Estate
YOUR_BUSINESS_NAME = os.getenv("YOUR_BUSINESS_NAME", "Prestige Properties")
APPOINTMENT_DURATION_MINUTES = 30
TIMEZONE = "America/New_York"  # Timezone of the properties/calendar

# --- NEW: Real Estate Property Details ---
# This is the data source for the agent's knowledge
PROPERTIES = [
    {
        "id": "sunset_villas",
        "name": "Sunset Villas",
        "type": "luxury apartment complex",
        "location": "downtown Metropolis",
        "description": "Offering stunning city views, a rooftop pool, and state-of-the-art amenities. We have 2 and 3 bedroom units available.",
        "talking_point": "It's perfect for professionals who want a vibrant city life."
    },
    {
        "id": "oakwood_estates",
        "name": "Oakwood Estates",
        "type": "gated community of single-family homes",
        "location": "the quiet suburbs of Greenwood",
        "description": "Spacious homes with large backyards, community parks, and excellent local schools.",
        "talking_point": "This is an ideal choice for growing families looking for a safe and peaceful neighborhood."
    }
]
