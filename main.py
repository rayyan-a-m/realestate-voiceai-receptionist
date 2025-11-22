import asyncio
import base64
import csv
import io
import json
import logging
import os
import warnings
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from google_auth_oauthlib.flow import Flow
from google.cloud import secretmanager
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

import config
from realtime.call_flow import CallFlowManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


# --- Initialization ---
app = FastAPI()

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
twilio_client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
active_call_flows: Dict[str, CallFlowManager] = {}
secret_manager_client = secretmanager.SecretManagerServiceClient()

# --- Application Lifecycle Events ---
@app.on_event("shutdown")
async def on_shutdown():
    """Gracefully stop all active call flows on server shutdown."""
    logging.info("Server is shutting down. Cleaning up active calls...")
    # Create a list of tasks to await
    cleanup_tasks = [
        call_flow.stop() for call_flow in active_call_flows.values()
    ]
    # Wait for all stop methods to complete
    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    active_call_flows.clear()
    logging.info("All active calls have been cleaned up.")


# ---------------------------------------------------------------------------
# Healthcheck endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root_healthcheck():
    """Basic health check to confirm the server is running."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Twilio call handling (Gemini Live)
# ---------------------------------------------------------------------------
@app.post("/inbound_call")
async def handle_inbound_call(request: Request):
    """
    Handles inbound calls from Twilio.
    This endpoint generates TwiML to connect the call to our WebSocket server.
    """
    logging.info("Inbound call received")
    
    # Determine WebSocket URL dynamically from the request headers
    host = request.headers.get("host", "localhost")
    # Use wss for secure WebSockets, which is required by Twilio
    websocket_url = f"wss://{host}/ws?call_type=INBOUND"
    
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=websocket_url)
    response.append(connect)
    
    logging.info(f"Generated TwiML for inbound call: {str(response)}")
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    name: str = "Valued Customer",
    call_type: str = "INBOUND"
):
    """
    Handles the bidirectional audio stream with Twilio via WebSocket.
    - Receives audio from Twilio.
    - Forwards audio to the Gemini Live API.
    - Receives audio from Gemini.
    - Forwards audio back to Twilio.
    """
    await websocket.accept()
    logging.info(f"WebSocket connection accepted. Lead name: {name}, Type: {call_type}")
    
    call_sid = "Unknown"
    stream_sid = "Unknown"
    call_flow: Optional[CallFlowManager] = None

    async def twilio_ws_send(message: Dict[str, Any]):
        """Helper to send a JSON message to the Twilio WebSocket."""
        await websocket.send_text(json.dumps(message))

    try:
        # The first message from Twilio is a 'connected' event.
        connected_message = await websocket.receive_text()
        connected_data = json.loads(connected_message)
        event = connected_data.get("event")

        if event != "connected":
            logging.error(f"Expected 'connected' event, but received '{event}'. Closing connection.")
            await websocket.close()
            return
        
        logging.info(f"Twilio 'connected' event received. Protocol: {connected_data.get('protocol')}, Version: {connected_data.get('version')}")

        # The second message from Twilio is a 'start' event, containing call details.
        start_message = await websocket.receive_text()
        start_data = json.loads(start_message)
        event = start_data.get("event")

        if event != "start":
            logging.error(f"Received unexpected event '{event}' before 'start'. Closing connection.")
            await websocket.close()
            return

        # Extract call and stream SIDs from the start message
        call_sid = start_data.get("start", {}).get("callSid", "Unknown")
        stream_sid = start_data.get("streamSid", "Unknown")
        logging.info("Twilio stream started: call=%s stream=%s type=%s", call_sid, stream_sid, call_type)

        # Initialize and start the call flow manager
        call_flow = CallFlowManager(
            call_sid=call_sid,
            stream_sid=stream_sid,
            twilio_ws_send_callback=twilio_ws_send,
        )
        active_call_flows[stream_sid] = call_flow
        await call_flow.start()

        # Main loop to process incoming messages from Twilio
        while True:
            message_str = await websocket.receive_text()
            payload = json.loads(message_str)
            event = payload.get("event")

            if event == "media" and call_flow:
                # Forward audio chunks to the call flow manager
                audio_bytes = base64.b64decode(payload["media"]["payload"])
                await call_flow.handle_audio_from_twilio(audio_bytes)
            elif event == "stop":
                # Twilio signals that the stream has ended
                logging.info("Twilio stop event received for stream %s.", stream_sid)
                break

    except WebSocketDisconnect:
        logging.info(f"WebSocket connection closed by client for call SID: {call_sid}")
    except Exception as e:
        logging.error(f"Error in WebSocket endpoint for call {call_sid}: {e}", exc_info=True)
    finally:
        # Cleanup resources
        if call_flow:
            await call_flow.stop()
        active_call_flows.pop(stream_sid, None)
        logging.info("Cleaned up resources for stream %s", stream_sid)


# --- Google Calendar OAuth 2.0 Endpoints ---

# NOTE: In a production environment, the REDIRECT_URI must be a public URL
# that you have registered in your Google Cloud Console for the OAuth client.
try:
    CLIENT_SECRETS_FILE = os.getenv("GOOGLE_OAUTH_WEB_CLIENT_SECRETS")
    if not CLIENT_SECRETS_FILE:
        raise ValueError("GOOGLE_OAUTH_WEB_CLIENT_SECRETS env var not set.")
    credentials_info = json.loads(CLIENT_SECRETS_FILE)
except (ValueError, json.JSONDecodeError) as e:
    logging.error(f"Error loading Google OAuth client secrets: {e}. Please check the environment variable.")
    credentials_info = None

# The redirect URI must match *exactly* one of the authorized redirect URIs
# for the OAuth 2.0 client, which you configure in the Google Cloud console.
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://realestate-voiceai-receptionist.onrender.com/oauth2callback")


@app.get("/auth", tags=["Google Calendar Auth"])
def auth(request: Request):
    """
    Generates the Google OAuth 2.0 authorization URL.
    Redirect the user to this URL to start the consent process.
    """
    if not credentials_info:
        return JSONResponse(status_code=500, content={"message": "OAuth client is not configured."})

    # The state parameter is used to prevent CSRF attacks.
    state = request.client.host
    flow = Flow.from_client_config(
        credentials_info,
        scopes=config.SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        # state=state # Optional: for CSRF protection
    )
    logging.info(f"Generated OAuth URL for {state}, redirecting to: {auth_url}")
    return RedirectResponse(auth_url)


@app.get("/oauth2callback", tags=["Google Calendar Auth"])
async def oauth2callback(request: Request):
    """
    Handles the callback from Google after the user grants consent.
    Fetches the OAuth 2.0 token and securely stores it in Google Secret Manager.
    """
    if not credentials_info:
        return JSONResponse(status_code=500, content={"message": "OAuth client is not configured."})

    # The full URL of the request is required to fetch the token.
    authorization_response = str(request.url)
    
    # For security, ensure the response is sent over HTTPS in production
    if "http://" in authorization_response and "localhost" not in authorization_response:
        logging.warning("OAuth callback received over HTTP. In production, this should be HTTPS.")
        # To enforce HTTPS, uncomment the following lines:
        # from fastapi import HTTPException
        # raise HTTPException(status_code=400, detail="OAuth callback must be over HTTPS")

    flow = Flow.from_client_config(
        credentials_info,
        scopes=config.SCOPES,
        redirect_uri=REDIRECT_URI
    )
    
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as e:
        logging.error(f"Failed to fetch OAuth token: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"message": "Failed to fetch OAuth token."})

    credentials = flow.credentials
    logging.info("OAuth token fetched successfully.")

    # --- Securely Store Client Credentials in Google Secret Manager ---
    try:
        # Get the user's email to use as a unique identifier for the secret.
        userinfo_response = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
        client_email = userinfo.get("email")

        if not client_email:
            logging.error("Could not retrieve user email from token.")
            return JSONResponse(status_code=400, content={"message": "Could not retrieve user email."})

        logging.info(f"Identified user for token storage: {client_email}")

        # Sanitize the email to create a valid Secret ID
        # (Secret IDs can only contain letters, numbers, hyphens, and underscores)
        secret_id = f"oauth-token-{client_email.replace('@', '-').replace('.', '-')}"
        
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
            logging.info(f"Checking if secret '{secret_id}' exists.")
            secret_manager_client.get_secret(request={"name": secret_name})
            logging.info(f"Secret '{secret_id}' already exists. Adding a new version.")
        except Exception: # google.api_core.exceptions.NotFound
            logging.info(f"Secret '{secret_id}' not found. Creating it now.")
            secret_manager_client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        
        # Add the token data as a new version of the secret
        secret_manager_client.add_secret_version(
            request={"parent": secret_name, "payload": {"data": payload}}
        )
        
        logging.info(f"Successfully saved token for '{client_email}' to Secret Manager as '{secret_id}'")

    except requests.HTTPError as http_err:
        logging.error(f"HTTP error while fetching user info: {http_err}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to fetch user info from Google."})
    except Exception as e:
        logging.error(f"Failed to save token to Secret Manager: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to save token to Secret Manager."})

    return {"message": "Google Calendar connected successfully and token stored securely!"}

# --- Outbound Campaign Management ---
outbound_leads_queue = asyncio.Queue()
campaign_in_progress = asyncio.Event()

async def campaign_worker(server_host: str):
    """Processes leads from the queue and initiates outbound calls."""
    logging.info("Starting outbound campaign worker...")
    campaign_in_progress.set()
    
    while not outbound_leads_queue.empty():
        lead = await outbound_leads_queue.get()
        first_name = lead.get('first_name', '').strip()
        phone_number = lead.get('phone', '').strip()
        
        if not phone_number:
            logging.warning(f"Skipping lead with missing phone number: {lead}")
            outbound_leads_queue.task_done()
            continue
            
        logging.info(f"Processing lead: {first_name} at {phone_number}")
        try:
            # Use the dynamically determined server host for the WebSocket URL
            websocket_url = f"wss://{server_host}/ws?name={first_name}&call_type=OUTBOUND"
            
            twiml = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            twiml.append(connect)
            
            logging.info(f"Initiating outbound call to {phone_number} with TwiML: {str(twiml)}")
            
            call = twilio_client.calls.create(
                to=phone_number,
                from_=config.TWILIO_PHONE_NUMBER,
                twiml=str(twiml)
            )
            logging.info(f"Outbound call initiated to {phone_number}, SID: {call.sid}")
            
            # Wait for a configurable duration before processing the next lead
            await asyncio.sleep(config.OUTBOUND_CALL_INTERVAL_SECONDS)
            
        except Exception as e:
            logging.error(f"Failed to call lead {first_name} at {phone_number}: {e}", exc_info=True)
        finally:
            outbound_leads_queue.task_done()
            
    logging.info("Outbound campaign finished.")
    campaign_in_progress.clear()

@app.post("/start_outbound_campaign")
async def start_outbound_campaign(request: Request, file: UploadFile = File(...)):
    """
    Starts an outbound calling campaign from a CSV file of leads.
    The CSV should have 'first_name' and 'phone' columns.
    """
    if campaign_in_progress.is_set():
        return JSONResponse(status_code=409, content={"status": "error", "message": "A campaign is already in progress."})
        
    logging.info("Received request to start outbound campaign.")
    
    try:
        # Read and parse the uploaded CSV file
        content = await file.read()
        file_data = io.StringIO(content.decode("utf-8-sig")) # Use utf-8-sig to handle potential BOM
        reader = csv.DictReader(file_data)
        
        leads = list(reader)
        if not leads:
            return JSONResponse(status_code=400, content={"status": "error", "message": "No leads found in the uploaded file."})

        for lead in leads:
            await outbound_leads_queue.put(lead)
            
        # Get the server host to construct the WebSocket URL
        server_host = request.headers.get("host", "localhost")
        
        # Start the campaign worker in the background
        asyncio.create_task(campaign_worker(server_host))
        
        message = f"Campaign started with {len(leads)} leads."
        logging.info(message)
        return {"status": "success", "message": message}
        
    except Exception as e:
        logging.error(f"Failed to process uploaded file or start campaign: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to process file."})

if __name__ == "__main__":
    logging.info("Starting FastAPI server with uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)