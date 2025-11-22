# Real Estate AI Voice Agent

## 1. Project Overview

This project is a sophisticated, real-time AI voice agent designed for a real estate firm. The agent's primary purpose is to handle inbound and outbound phone calls, qualify potential leads, check for available property viewing times, and book appointments directly on a Google Calendar. It uses a combination of Twilio for telephony, Google's Gemini Live for real-time conversation, and Google Calendar for scheduling.

The agent, named 'Sky', is designed to be professional, engaging, and efficient, providing a seamless experience for clients looking to schedule property viewings.

## 2. Architecture

The application is built using Python and the FastAPI framework, orchestrating several cloud services to achieve real-time voice interaction.

-   **Web Server (`main.py`)**: A FastAPI server acts as the central hub. It exposes endpoints to handle webhooks from Twilio and manages the lifecycle of each call.
-   **Telephony (Twilio)**: Twilio is used to manage inbound and outbound calls. When a call is received, Twilio connects to the FastAPI server via a WebSocket, streaming the call audio in real-time.
-   **Real-time Conversation (Gemini Live)**: The core of the agent's intelligence lies in Google's Gemini Live.
    -   The application establishes a WebSocket connection with the Gemini Live API (`realtime/gemini_live_client.py`).
    -   Audio from the caller (streamed via Twilio) is forwarded directly to Gemini.
    -   Gemini handles speech-to-text, language understanding, and text-to-speech in a single, low-latency stream. This means the agent can think and respond while the user is still speaking.
    -   The agent's responses (as audio) are streamed back from Gemini, then forwarded to Twilio to be played to the caller.
-   **Agent Capabilities (Tools)**: The agent's ability to interact with the outside world is defined by a set of "tools". These are Python functions that the Gemini model can decide to call. The current tools are for Google Calendar integration (`google_calendar.py`):
    -   `find_available_slots`: Checks the calendar for open appointment times on a given day.
    -   `book_appointment`: Creates a new event in the calendar to book a property viewing.
-   **Agent Persona & Logic (`prompts.py`)**: The agent's personality, rules of engagement, and conversational flow are defined in a detailed system prompt (`SYSTEM_PROMPT`). This prompt instructs the agent on how to greet callers, what information to gather (name, email), when to use its tools, and how to close the conversation.
-   **Configuration (`config.py`)**: All sensitive information and settings (API keys, phone numbers, property details) are managed via environment variables, which are loaded into the application by this file.

### Call Flow (Inbound Example)

1.  A potential client calls the Twilio phone number.
2.  Twilio sends an HTTP request to the `/inbound_call` endpoint in `main.py`.
3.  The FastAPI server responds with TwiML instructions to open a WebSocket connection to its `/ws` endpoint.
4.  The `CallFlowManager` (`realtime/call_flow.py`) is instantiated for the new call.
5.  The `CallFlowManager` creates a `GeminiLiveClient` instance, which connects to the Google Gemini Live API.
6.  Audio from the caller is streamed: `Twilio -> FastAPI -> GeminiLiveClient -> Gemini Live API`.
7.  The Gemini model processes the audio, understands the user's intent, and decides on a response or a tool to use.
8.  If a tool is needed (e.g., `find_available_slots`), Gemini sends a request to the FastAPI server. The server executes the corresponding Python function in `google_calendar.py` and sends the result back to Gemini.
9.  Gemini generates a spoken response.
10. Audio from Gemini is streamed back: `Gemini Live API -> GeminiLiveClient -> FastAPI -> Twilio -> Caller`.
11. When the call ends, all WebSocket connections are closed.

## 3. Key Files

-   `main.py`: The main FastAPI application. It handles Twilio webhooks, manages WebSocket connections for audio streaming, and orchestrates the entire call flow. It also includes endpoints for Google OAuth2 and for starting outbound calling campaigns.
-   `google_calendar.py`: Contains the LangChain tools for interacting with the Google Calendar API (`find_available_slots`, `book_appointment`). It also manages Google OAuth2 authentication, using `token.json` or Google Secret Manager to store credentials.
-   `prompts.py`: Defines the `SYSTEM_PROMPT` for the AI agent. This is the most critical file for shaping the agent's behavior, goals, and personality. It explicitly outlines the conversation flow and when to use tools.
-   `config.py`: Loads all environment variables from a `.env` file and contains static configuration like property details and business information.
-   `realtime/gemini_live_client.py`: A wrapper class that manages the connection and communication with the Google Gemini Live streaming API. It handles sending audio from the user and receiving audio/text/tool-calls from the model.
-   `realtime/call_flow.py`: A manager class that ties everything together for a single call. It initializes the `GeminiLiveClient` and orchestrates the flow of data between Twilio and Gemini.
-   `local_test.py`: A valuable utility script for testing the agent without needing Twilio. It uses the local microphone and speakers to simulate a phone call, allowing for rapid testing of the STT -> LLM -> TTS loop.

## 4. Setup and Installation

1.  **Prerequisites**: Python 3.8+ and a Google Cloud project with the Calendar API and Vertex AI API enabled.
2.  **Clone the repository**.
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration**:
    -   Create a `.env` file in the root directory.
    -   Populate it with the necessary credentials and configuration (see `config.py` for required variables). This includes:
        -   `GOOGLE_CREDENTIALS_JSON`: The content of your Google Cloud service account JSON file.
        -   `GOOGLE_OAUTH_WEB_CLIENT_SECRETS`: The content of your OAuth 2.0 Client ID JSON file from Google Cloud.
        -   `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`.
        -   `GCP_PROJECT_ID`.
5.  **Google Calendar Authentication**:
    -   The first time you run the app or `google_calendar.py`, you may need to go through an OAuth flow in your browser to grant calendar access. This will create a `token.json` file. For production, the app is designed to store these tokens securely in Google Secret Manager.

## 5. Usage

### Running the Main Application

The application is a FastAPI server. To run it for development:

```bash
uvicorn main:app --reload --port 8000
```

For the Twilio integration to work, the server must be accessible from the public internet. Use a tool like `ngrok` during development to expose your local server.

### Local Testing (without Twilio)

To test the full conversational loop using your computer's microphone and speakers, run:

```bash
python local_test.py
```

This is the recommended way to test changes to the agent's prompt or tools quickly.

## 6. How to Modify the Agent

-   **To change the agent's personality, instructions, or knowledge**: Modify the `SYSTEM_PROMPT` string in `prompts.py`. This is the primary control panel for the agent's behavior.
-   **To add new capabilities (e.g., CRM integration)**:
    1.  Create a new function and decorate it with `@tool` from LangChain (e.g., in `google_calendar.py` or a new file).
    2.  Add the new tool to the list of tools available to the agent.
    3.  Update the `SYSTEM_PROMPT` in `prompts.py` to teach the agent how and when to use its new tool.
-   **To change the agent's voice**: Modify the `GEMINI_VOICE_NAME` in `config.py`.
-   **To change the underlying LLM**: Update the `GEMINI_LIVE_MODEL` variable in `config.py`.
-   **To add or update property listings**: Modify the `PROPERTIES` list in `config.py`.
