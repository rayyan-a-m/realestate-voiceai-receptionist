# AI Voice Agent for Real Estate - Copilot Instructions

This document provides guidance for AI agents working on the Real Estate AI Voice Agent project.

## 1. Project Overview & Architecture

This project is a real-time AI voice agent for a real estate firm that handles calls, qualifies leads, and books appointments on Google Calendar.

- **Core Logic**: The main application is in `main.py`, a FastAPI server that orchestrates all services.
- **Telephony**: Twilio handles inbound and outbound calls. The `/inbound_call` endpoint initiates a WebSocket connection for real-time audio streaming.
- **Speech-to-Text (STT)**: Google Cloud Speech-to-Text is used for real-time transcription. See the `transcription_agent_task` function in `main.py`.
- **Text-to-Speech (TTS)**: Google Cloud Text-to-Speech generates the agent's voice. See `generate_and_stream_audio` in `main.py`.
- **The "Brain"**: The agent's intelligence comes from a LangChain agent using Google's Gemini model (`ChatVertexAI` in `main.py`).
- **Agent's Capabilities (Tools)**: The agent can interact with Google Calendar using tools defined in `google_calendar.py`. These are `find_available_slots` and `book_appointment`.
- **Agent's Personality & Goals**: The core prompt defining the agent's persona, rules, and conversation flow is located in `prompts.py`.

## 2. Key Files

- `main.py`: The central FastAPI application. Handles Twilio webhooks, WebSocket audio streaming, and orchestrates the STT, LLM, and TTS pipeline.
- `google_calendar.py`: Contains the LangChain tools for interacting with the Google Calendar API. This is how the agent checks for availability and books appointments. It also handles Google OAuth2 authentication.
- `prompts.py`: Defines the system prompt for the LangChain agent. This is the most critical file for defining the agent's behavior, goals, and personality.
- `config.py`: Loads all environment variables and contains static configuration like property details.
- `requirements.txt`: Lists all Python dependencies.
- `llm_test.py`: A script to test the connection to the Vertex AI LLM.
- `local_test.py`: A script for testing the voice agent locally using your microphone and speakers, bypassing Twilio.

## 3. Developer Workflows

### Running the Main Application

The application is a FastAPI server. To run it for development:

```bash
uvicorn main:app --reload --port 8000
```

For production, you'll need a public URL (e.g., using ngrok or a cloud deployment) for Twilio webhooks to work.

### Testing

- **LLM Connection Test**: To ensure the Gemini model is accessible, run:
  ```bash
  python llm_test.py
  ```
- **Local Voice Test**: To test the full STT -> LLM -> TTS loop without Twilio, run:
  ```bash
  python local_test.py
  ```
  This will use your local microphone.

### Google Authentication

- The first time you run the application or `google_calendar.py`, you will be taken through an OAuth2 flow in your browser to grant access to your Google Calendar. This will create a `token.json` file.
- You must have a `credentials.json` file from Google Cloud with the Calendar API enabled.

## 4. How to Modify the Agent

- **To change the agent's personality, instructions, or knowledge of properties**: Modify the `SYSTEM_PROMPT` in `prompts.py`.
- **To add new capabilities (e.g., CRM integration)**:
  1. Create a new function in `google_calendar.py` or a new file.
  2. Decorate it with `@tool` from LangChain.
  3. Add the new tool to the `tools` list in `main.py`.
  4. Update the `SYSTEM_PROMPT` in `prompts.py` to teach the agent how and when to use the new tool.
- **To change the voice**: Modify the `VoiceSelectionParams` in `generate_and_stream_audio` in `main.py`.
- **To change the LLM model**: Update the `ChatVertexAI` initialization in `main.py`.
