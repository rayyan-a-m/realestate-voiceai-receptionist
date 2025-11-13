import asyncio
from queue import Queue # Use standard queue for thread-safe operations
import os
import base64
import json
import csv
import io
import logging
import warnings
from urllib.parse import urlparse
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tame noisy third-party warnings in runtime logs
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"vertexai\.generative_models\._generative_models",
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"langchain\.agents\.json_chat\.base",
)

import pytz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
import uvicorn
import datetime
from google_auth_oauthlib.flow import Flow

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from vertexai import agent_engines
import vertexai  # Needed for explicit project/location initialization


# --- Agent Wrapper for Error Handling ---
# class SafeLangchainAgent(agent_engines.LangchainAgent):
#     """
#     A wrapper around LangchainAgent to gracefully handle internal errors.

#     The base LangchainAgent can raise an AttributeError if the underlying model
#     returns an error payload (often a list) instead of a valid JSON response.
#     This wrapper catches that specific error during the query and returns a
#     structured error message, preventing the main application from crashing.
#     """
#     def query(self, *args, **kwargs):
#         try:
#             return super().query(*args, **kwargs)
#         except AttributeError as e:
#             # This error is typically raised when the agent's internal parser
#             # receives an error list from the model instead of a dict.
#             logging.error(f"Caught AttributeError inside LangchainAgent query: {e}")
#             # Return a consistent error format that the rest of the app can handle.
#             return {
#                 "output": "I'm sorry, I encountered a processing error. Could you please try again?"
#             }


from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import secretmanager
import requests

import config
from prompts import prompt, SYSTEM_PROMPT
from google_calendar import find_available_slots, book_appointment

# --- Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google Cloud & Vertex AI Init ---
# Centralized credentials from config.py
google_credentials = config.GOOGLE_CREDENTIALS 

if not config.GCP_PROJECT_ID:
    logging.error("GCP_PROJECT_ID is not set. Please check your .env file or environment variables.")
else:
    try:
        vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION, credentials=google_credentials)
        logging.info(f"Vertex AI initialized for project '{config.GCP_PROJECT_ID}' in location '{config.GCP_LOCATION}'.")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}")

# --- Service Clients ---
twilio_client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
speech_client = speech.SpeechClient(credentials=google_credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=google_credentials)

logging.info("Service clients initialized.")

# --- LangChain Agent Setup ---
# A dictionary to store chat histories. In a production environment,
# you would use a more persistent storage solution.
session_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieves a chat history for a given session ID, creating a new one if it
    doesn't exist.
    """
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

try:
    if not config.GCP_PROJECT_ID:
        raise RuntimeError("Missing GCP_PROJECT_ID; cannot create LangchainAgent.")
    
    agent = agent_engines.LangchainAgent(
        model=config.VERTEX_MODEL,
        tools=[find_available_slots, book_appointment],
        model_kwargs={
                "temperature": 0.0,
                "max_output_tokens": 256,
                "top_p": 0.95,
            },
        system_instruction=SYSTEM_PROMPT,
        chat_history=get_session_history
    )
    logging.info(f"Vertex AI LangchainAgent initialized successfully with model '{config.VERTEX_MODEL}'.")
except Exception as e:
    logging.error(f"Failed to initialize LLM Agent: {e}", exc_info=True)
    # Depending on the severity, you might want to exit
    # raise SystemExit(1) from e



def _to_text(content) -> str:
    """Normalize agent/LLM content to plain string for TTS.

    Tries common shapes returned by Vertex AI + LangChain across versions:
    - Plain string
    - Objects with .text / .output_text / .content
    - Dicts with keys like ["output", "text", "content", "result", "message"]
    - Lists of the above (flattens recursively)
    Never raises on unexpected shapes; always returns a best-effort string.
    """
    try:
        logging.info(f"Normalizing content to text: {content} (type: {type(content)})")
        if content is None:
            return ""

        # Handle lists first, as they are a common error format from Vertex
        if isinstance(content, list):
            parts = [
                _to_text(part)
                for part in content
                if part is not None and _to_text(part) is not None
            ]
            # Filter out empties while preserving spacing
            return " ".join(p for p in parts if p)

        # Common simple cases
        if isinstance(content, str):
            return content

        # Vertex / LangChain objects often expose .text or .output_text
        for attr in ("text", "output_text"):
            if hasattr(content, attr):
                try:
                    val = getattr(content, attr)
                    return val if isinstance(val, str) else _to_text(val)
                except Exception:
                    pass

        # Some responses expose a .content attribute which may itself be a list or nested object
        if hasattr(content, "content"):
            try:
                return _to_text(getattr(content, "content"))
            except Exception:
                pass

        # Dict-like structures: check a few likely fields
        if isinstance(content, dict):
            for key in ("output", "text", "content", "result", "message", "response"):
                if key in content and content[key]:
                    return _to_text(content[key])
            # Some Vertex responses contain "candidates" -> [ {"content": ...} ]
            candidates = content.get("candidates") if "candidates" in content else None
            if candidates:
                return _to_text(candidates)
            # Fallback to stringifying the dict
            return str(content)

        # Anything else: best-effort string
        return str(content)
    except Exception:
        # Absolute fallback: never propagate parsing errors to the call site
        try:
            return str(content)
        except Exception:
            return ""

# --- Healthcheck & Root ---
@app.get("/")
async def root_healthcheck():
    """Simple healthcheck endpoint for Render and uptime monitors."""
    return JSONResponse({"status": "ok"})


# --- Real-time Transcription & Agent Logic ---

async def send_clear_message(websocket: WebSocket, stream_sid: str):
    """Sends a clear message to Twilio to stop any queued audio."""
    try:
        clear_message = {
            "event": "clear",
            "streamSid": stream_sid,
        }
        await websocket.send_text(json.dumps(clear_message))
        logging.info(f"Sent clear message to Twilio for stream {stream_sid}")
    except WebSocketDisconnect:
        logging.warning(f"Failed to send clear message: WebSocket for stream {stream_sid} is disconnected.")
    except Exception as e:
        logging.error(f"Error sending clear message for stream {stream_sid}: {e}")

async def generate_and_stream_audio(text: str, websocket: WebSocket, stream_sid: str, interruption_event: asyncio.Event):
    """Generates audio using Google TTS and streams it to Twilio, checking for interruptions."""
    if not text or not text.strip():
        logging.warning("TTS text is empty; skipping audio generation.")
        return False

    logging.info(f"Generating audio for stream {stream_sid}: '{text}'")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Standard-C"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MULAW,
            sample_rate_hertz=8000
        )

        response = await asyncio.to_thread(
            tts_client.synthesize_speech,
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Send audio in chunks to allow for interruption
        chunk_size = 1600 # 100ms of 8kHz 8-bit audio
        for i in range(0, len(response.audio_content), chunk_size):
            if interruption_event.is_set():
                logging.info(f"Interruption detected. Stopping audio stream {stream_sid}.")
                return True # Interrupted

            chunk = response.audio_content[i:i + chunk_size]
            encoded_chunk = base64.b64encode(chunk).decode('utf-8')
            media_message = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": encoded_chunk}
            }
            await websocket.send_text(json.dumps(media_message))
            # Small sleep to allow interruption event to be processed
            await asyncio.sleep(0.05)

        logging.info(f"Finished streaming audio for stream {stream_sid}")
        return False # Not interrupted

    except Exception as e:
        logging.error(f"Error generating or streaming audio for stream {stream_sid}: {e}", exc_info=True)
        return False

async def handle_agent_response(transcript: str, call_sid: str, stream_sid: str, websocket: WebSocket):
    """Gets response from LLM, synthesizes audio, and streams it back to Twilio."""
    interruption_event = asyncio.Event()

    # The agent task will run in the background
    async def agent_task():
        logging.info(f"Querying agent for call {call_sid} with: '{transcript}'")
        try:
            # Use the global agent instance
            logging.info(f"starting Querying agent for call {call_sid}")
            agent_response = await asyncio.to_thread(
                agent.query,
                input=transcript,
                config={"configurable": {"session_id": call_sid}},
            )
            logging.info(f"Agent response for call {call_sid}: {agent_response}")

            # The agent may return a list on error. If so, handle it as a failure.
            if isinstance(agent_response, list):
                logging.error(f"Agent returned a list, treating as error. Payload: {agent_response}")
                raise AttributeError(f"Agent returned a list: {agent_response}")

            tts_text = _to_text(agent_response)
            logging.info(f"Normalized agent response to text: '{tts_text}'")
            if not tts_text or not tts_text.strip():
                # Try a few more common fields/paths defensively
                fallback = None
                if isinstance(agent_response, dict):
                    fallback = agent_response.get("output") or agent_response.get("text")
                if not fallback:
                    fallback = str(agent_response)
                tts_text = _to_text(fallback)
        except AttributeError as e:
            # Common when underlying Google client returns a list payload on error
            logging.error(f"AttributeError during agent response (likely malformed error payload): {e}")
            tts_text = (
                "I'm sorry, I had trouble processing that. Could you please repeat, or let me know if you'd like to book a property visit?"
            )
        except Exception as e:
            logging.error(f"Error in agent task for call {call_sid}: {e}", exc_info=True)
            tts_text = (
                "I ran into a technical issue. Could you restate that, or tell me a good time you'd like to visit a property?"
            )

        logging.info(f"Agent for call {call_sid} says: {tts_text}")

        was_interrupted = await generate_and_stream_audio(tts_text, websocket, stream_sid, interruption_event)

        if was_interrupted:
            await send_clear_message(websocket, stream_sid)
            logging.info(f"Agent speech for {call_sid} was interrupted.")

    # Return the task and the interruption event so the main loop can manage it
    return asyncio.create_task(agent_task()), interruption_event

def get_stt_config():
    """Returns the Google STT streaming configuration for Twilio."""
    return speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            language_code="en-US",
            model="telephony",
            use_enhanced=True,
            enable_automatic_punctuation=True,
        ),
        single_utterance=True, # Critical for turn-based conversation
        interim_results=False,
    )

async def transcription_agent_task(websocket: WebSocket, call_sid: str, stream_sid: str, lead_name: str | None = None):
    """Handles the full lifecycle of a voice call using Google STT/TTS with interruption."""
    logging.info(f"Starting agent task for call {call_sid}, stream {stream_sid}, name: {lead_name}")
    
    main_loop = asyncio.get_running_loop()
    audio_queue = Queue()
    stop_event = asyncio.Event()
    active_agent_task = None
    interruption_event = None

    # --- Audio Receiver (Async) ---
    async def audio_receiver():
        """Receives audio from Twilio and puts it into a standard queue."""
        logging.info(f"Audio receiver started for {call_sid}")
        try:
            while not stop_event.is_set():
                message_str = await websocket.receive_text()
                data = json.loads(message_str)
                event = data.get("event")

                if event == "media":
                    payload = data["media"]["payload"]
                    audio_queue.put(base64.b64decode(payload))
                elif event == "start":
                     # This is handled in the main websocket_endpoint now
                    pass
                elif event == "stop":
                    logging.info(f"Twilio 'stop' event received for call {call_sid}")
                    stop_event.set()
                    break
        except WebSocketDisconnect:
            logging.warning(f"Twilio WebSocket disconnected for call {call_sid}")
        except Exception as e:
            logging.error(f"Error in audio_receiver for {call_sid}: {e}")
        finally:
            audio_queue.put(None) # Signal the end of audio
            logging.info(f"Audio receiver stopped for {call_sid}")

    # --- STT Worker (Sync, runs in a separate thread) ---
    def stt_worker(loop):
        """Processes audio from the queue using Google STT in a blocking manner."""
        
        def audio_generator():
            """Yields audio chunks from the queue for the STT client."""
            while not stop_event.is_set():
                chunk = audio_queue.get()
                if chunk is None:
                    return
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
                audio_queue.task_done()

        try:
            stt_config = get_stt_config()
            responses = speech_client.streaming_recognize(
                config=stt_config,
                requests=audio_generator(),
            )

            for response in responses:
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                
                transcript = result.alternatives[0].transcript.strip()

                if result.is_final and transcript:
                    logging.info(f"Final transcript for {call_sid}: '{transcript}'")
                    # Since this is a sync function, we need to run the async handler
                    # in the main event loop.
                    asyncio.run_coroutine_threadsafe(
                        handle_final_transcript(transcript),
                        loop
                    )
                    # Because single_utterance=True, the stream will close here.
                    # We break to allow the worker to exit and be restarted for the next turn.
                    break 

        except Exception as e:
            if "DEADLINE_EXCEEDED" in str(e):
                logging.warning(f"STT stream deadline exceeded for {call_sid}. This is expected on silence.")
            else:
                logging.error(f"STT worker error for {call_sid}: {e}", exc_info=True)
        finally:
            logging.info(f"STT worker finished for {call_sid}.")
            # Do not set stop_event here, let the main loop control it

    # --- Final Transcript Handler (Async) ---
    async def handle_final_transcript(transcript: str):
        """Handles the logic for when a final transcript is received."""
        nonlocal active_agent_task, interruption_event
        try:
            # If an agent is currently speaking, interrupt it.
            if active_agent_task and interruption_event and not interruption_event.is_set():
                logging.info(f"User interrupted. Setting interruption event for {call_sid}.")
                interruption_event.set()
                await active_agent_task  # Wait for the interrupted task to clean up

            # Start a new agent response task
            active_agent_task, interruption_event = await handle_agent_response(
                transcript, call_sid, stream_sid, websocket
            )
        except Exception as e:
            logging.error(f"Error handling final transcript for {call_sid}: {e}", exc_info=True)


    # --- Main Task Logic ---
    receiver_task = asyncio.create_task(audio_receiver())

    try:
        # 1. Greet the user
        greeting_text = f"Hi, am I speaking with {lead_name}?"
        interruption_event = asyncio.Event() # Initial event for the greeting
        await generate_and_stream_audio(greeting_text, websocket, stream_sid, interruption_event)

        # 2. Start the STT worker loop
        while not stop_event.is_set():
            # Run the synchronous STT worker in a background thread
            await asyncio.to_thread(stt_worker, main_loop)
            # After stt_worker finishes (due to single_utterance), it will loop
            # and start a new recognition stream, unless the stop_event is set.
            if stop_event.is_set():
                break
            # Small sleep to prevent tight looping if something goes wrong
            await asyncio.sleep(0.1)

    except Exception as e:
        logging.error(f"Error in transcription_agent_task for {call_sid}: {e}", exc_info=True)
    finally:
        stop_event.set()
        if receiver_task:
            await receiver_task
        if active_agent_task:
            await active_agent_task
        logging.info(f"Agent task finished for call {call_sid}.")


# --- FastAPI Endpoints ---
@app.post("/inbound_call")
async def handle_inbound_call():
    logging.info("Inbound call received")
    response = VoiceResponse()
    connect = Connect()
    # Ensure the websocket URL is correct for your deployment
    connect.stream(url=f"wss://realestate-voiceai-receptionist.onrender.com/ws?type=INBOUND")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str = "z", type: str = "INBOUND"):
    """Handles the WebSocket connection from Twilio."""
    await websocket.accept()

    # ✅ Parse 'name' from the query string manually
    query_params = dict(websocket.query_params)
    call_type = query_params.get("type", "INBOUND")
    if call_type == "OUTBOUND":
        name = query_params.get("name", "z")

    logging.info(f"WebSocket connection accepted. Lead name: {name}, Type: {call_type}")
    call_sid = "Unknown"
    stream_sid = "Unknown"
    
    try:
        # First message is 'connected'
        connected_message = await websocket.receive_text()
        connected_data = json.loads(connected_message)
        event = connected_data.get("event")

        if event != "connected":
            logging.error(f"Expected 'connected' event, but received '{event}'. Closing connection.")
            await websocket.close()
            return
        
        logging.info(f"Twilio 'connected' event received. Protocol: {connected_data.get('protocol')}, Version: {connected_data.get('version')}")

        # The second message from Twilio should be a 'start' event
        start_message = await websocket.receive_text()
        start_data = json.loads(start_message)
        event = start_data.get("event")

        if event != "start":
            logging.error(f"Received unexpected event '{event}' before 'start'.")
            await websocket.close()
            return

        # Extract call and stream SIDs from the start message
        call_sid = start_data.get("start", {}).get("callSid", "Unknown")
        stream_sid = start_data.get("streamSid", "Unknown")
        logging.info(f"Twilio 'start' event received for call {call_sid}, stream {stream_sid}")

        # Start the main transcription and agent processing task
        await transcription_agent_task(websocket, call_sid, stream_sid, name)

    except WebSocketDisconnect:
        logging.info(f"WebSocket connection closed for call SID: {call_sid}")
    except Exception as e:
        logging.error(f"Error in WebSocket endpoint for call {call_sid}: {e}", exc_info=True)
    finally:
        logging.info(f"WebSocket endpoint closing for call SID: {call_sid}.")
        if call_sid in session_histories:
            del session_histories[call_sid]
            logging.info(f"Cleaned up chat history for call {call_sid}.")


# --- Google Calendar OAuth 2.0 Endpoints ---

# NOTE: In a production environment, the REDIRECT_URI must be a public URL
# that you have registered in your Google Cloud Console for the OAuth client.
# For local testing, you can use a tool like ngrok to expose your localhost.
CLIENT_SECRETS_FILE = "credentials.json"
# The redirect URI must match *exactly* one of the authorized redirect URIs
# for the OAuth 2.0 client, which you configure in the Google Cloud console.
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://realestate-voiceai-receptionist.onrender.com/oauth2callback")


# CLIENT URL
@app.get("/auth", tags=["Google Calendar Auth"])
def auth(request: Request):
    """
    Generates the Google OAuth 2.0 authorization URL.
    Redirect the user to this URL to start the consent process.
    """
    # The state parameter is used to prevent CSRF attacks.
    # You can use it to store session-specific information.
    # Here we use the client's host as a simple state.
    state = request.client.host
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=config.SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        # You can pass the state here if you need to
        # state=state
    )
    logging.info(f"Generated OAuth URL for {request.client.host}: {auth_url}")
    return {"auth_url": auth_url}


@app.get("/oauth2callback", tags=["Google Calendar Auth"])
async def oauth2callback(request: Request):
    """
    Handles the callback from Google after the user grants consent.
    Fetches the OAuth 2.0 token and saves it for the client.
    """
    # The full URL of the request is required to fetch the token.
    authorization_response = str(request.url)
    
    # For security, ensure the response is sent over HTTPS in production
    if "http://" in authorization_response and "localhost" not in authorization_response:
        logging.warning("OAuth callback received over HTTP. In production, this should be HTTPS.")
        # In a production environment, you might want to enforce HTTPS:
        # raise HTTPException(status_code=400, detail="OAuth callback must be over HTTPS")

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=config.SCOPES,
        redirect_uri=REDIRECT_URI
    )
    
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as e:
        logging.error(f"Failed to fetch OAuth token: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"message": "Failed to fetch OAuth token."})

    credentials = flow.credentials

    logging.info(f"OAuth token fetched successfully for client - {credentials}")

    # --- Securely Store Client Credentials in Google Secret Manager ---
    try:
        # Initialize Secret Manager client
        secret_client = secretmanager.SecretManagerServiceClient(credentials=google_credentials)
        
        # Use the user's email as a unique identifier.
        # First, get the user's info using the access token.
        userinfo_response = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        )
        userinfo = userinfo_response.json()
        client_email = userinfo.get("email")

        if not client_email:
            logging.error("Could not retrieve user email from token.")
            return JSONResponse(status_code=400, content={"message": "Could not retrieve user email."})

        logging.info(f"Identified user for token storage: {client_email}")

        # Sanitize the email to create a valid Secret ID
        # (Secret IDs can only contain letters, numbers, hyphens, and underscores)
        secret_id = f"oauth-token-{client_email.replace('@', '-').replace('.', '-')}"
        
        # The parent project for the secret
        parent = f"projects/{config.GCP_PROJECT_ID}"
        secret_name = f"{parent}/secrets/{secret_id}"

        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }
        payload = json.dumps(token_data).encode("UTF-8")

        # Check if the secret exists. If not, create it.
        try:
            secret_client.get_secret(request={"name": secret_name})
            logging.info(f"Secret '{secret_id}' already exists. Adding a new version.")
        except Exception: # google.api_core.exceptions.NotFound
            logging.info(f"Secret '{secret_id}' not found. Creating it now.")
            secret_client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        
        # Add the token data as a new version of the secret
        secret_client.add_secret_version(
            request={"parent": secret_name, "payload": {"data": payload}}
        )
        
        logging.info(f"Successfully saved token for client '{client_email}' to Secret Manager as '{secret_id}'")

    except Exception as e:
        logging.error(f"Failed to save token to Secret Manager: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to save token to Secret Manager."})

    return {"message": "Google Calendar connected successfully and token stored securely!"}

# --- Outbound Campaign Management ---
outbound_leads_queue = asyncio.Queue()
campaign_in_progress = False

async def campaign_worker():
    global campaign_in_progress
    campaign_in_progress = True
    logging.info("Starting outbound campaign worker...")
    while not outbound_leads_queue.empty():
        lead = await outbound_leads_queue.get()
        logging.info(f"Processing lead: {lead['first_name']} {lead['last_name']} at {lead['phone']}")
        try:
            # Note: Update wss URL to your deployed server's URL
            websocket_url = f"wss://realestate-voiceai-receptionist.onrender.com/ws?name={lead['first_name']}&type=OUTBOUND"
            twiml_response = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            twiml_response.append(connect)
            logging.info(f"Initiating outbound call to {lead['phone']} with TwiML: {str(twiml_response)}")
            call = twilio_client.calls.create(to=lead['phone'], from_=config.TWILIO_PHONE_NUMBER, twiml=str(twiml_response))
            logging.info(f"Outbound call initiated to {lead['phone']}, SID: {call.sid}")
            await asyncio.sleep(15) # Wait before processing the next lead
        except Exception as e:
            logging.error(f"Failed to call lead {lead['first_name']}: {e}", exc_info=True)
        outbound_leads_queue.task_done()
    logging.info("Outbound campaign finished.")
    campaign_in_progress = False

@app.post("/start_outbound_campaign")
async def start_outbound_campaign(file: UploadFile = File(...)):
    global campaign_in_progress
    if campaign_in_progress:
        return {"status": "error", "message": "A campaign is already in progress."}
    logging.info("Received request to start outbound campaign.")
    try:
        # Reaading file content from csv
        content = await file.read()
        file_data = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(file_data)
        leads_loaded = 0
        for row in reader:
            await outbound_leads_queue.put(row)
            leads_loaded += 1
        if leads_loaded > 0:
            asyncio.create_task(campaign_worker())
            message = f"Campaign started with {leads_loaded} leads."
        else:
            message = "No leads found in the uploaded file."
        logging.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        logging.error(f"Failed to process uploaded file: {e}", exc_info=True)
        return {"status": "error", "message": "Failed to process file."}

if __name__ == "__main__":
    logging.info("Starting server with uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=8000)
