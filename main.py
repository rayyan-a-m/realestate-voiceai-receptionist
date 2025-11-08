import asyncio
from asyncio import Queue
import os
import base64
import json
import csv
import io
import logging
from urllib.parse import urlparse
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pytz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
import uvicorn
import datetime

from langchain_core.messages import HumanMessage, AIMessage
from vertexai import agent_engines

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as texttospeech

import config
from prompts import prompt
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

twilio_client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

logging.info("Service initialized")

# --- LangChain Agent Setup (inspired by local_test.py) ---
try:
    agent = agent_engines.LangchainAgent(
        model="gemini-1.5-flash-lite", # Using a recommended model
        tools=[find_available_slots, book_appointment],
        model_kwargs={
            "temperature": 0.0,
            "max_output_tokens": 512, # Increased for potentially complex tool responses
            "top_p": 0.95,
        },
        # System instructions can be passed here if needed
        # system_instruction=prompt.messages[0].prompt.template
    )
    logging.info("Vertex AI LangchainAgent initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize LLM Agent: {e}", exc_info=True)
    # Exit if the agent fails to initialize, as it's critical
    exit(1)


def _to_text(content) -> str:
    """Normalize agent/LLM content to plain string for TTS."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("output", "")
    if hasattr(content, "content"):
        return _to_text(content.content)
    if isinstance(content, list):
        return " ".join(_to_text(part) for part in content if part)
    return str(content)

# This dictionary will hold the chat history for each active call
conversation_history = {}

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
            websocket_url = f"wss://realestate-voiceai-receptionist.onrender.com/ws?name={lead['first_name']}"
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
            agent_response = await asyncio.to_thread(agent.query, input=transcript)
            tts_text = _to_text(agent_response)

            logging.info(f"Agent for call {call_sid} says: {tts_text}")
            
            # Update chat history
            chat_history = conversation_history.get(call_sid, [])
            chat_history.append(HumanMessage(content=transcript))
            chat_history.append(AIMessage(content=tts_text))
            conversation_history[call_sid] = chat_history

            was_interrupted = await generate_and_stream_audio(tts_text, websocket, stream_sid, interruption_event)

            if was_interrupted:
                await send_clear_message(websocket, stream_sid)
                logging.info(f"Agent speech for {call_sid} was interrupted.")

        except Exception as e:
            logging.error(f"Error in agent task for call {call_sid}: {e}", exc_info=True)

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
    logging.info(f"Starting agent task for call {call_sid}, stream {stream_sid}")
    
    audio_queue = Queue()
    stop_event = asyncio.Event()
    active_agent_task = None
    interruption_event = None

    async def audio_receiver():
        """Receives audio from Twilio and puts it into a queue."""
        logging.info(f"Audio receiver started for {call_sid}")
        try:
            while not stop_event.is_set():
                message_str = await websocket.receive_text()
                data = json.loads(message_str)
                event = data.get("event")
                if event == "media":
                    payload = data["media"]["payload"]
                    await audio_queue.put(base64.b64decode(payload))
                elif event == "stop":
                    logging.info(f"Twilio stop event received for call {call_sid}")
                    stop_event.set()
                    break
        except WebSocketDisconnect:
            logging.warning(f"Twilio WebSocket disconnected for call {call_sid}")
        except Exception as e:
            logging.error(f"Error in audio_receiver for {call_sid}: {e}")
        finally:
            stop_event.set()
            await audio_queue.put(None) # Signal end of audio
            logging.info(f"Audio receiver stopped for {call_sid}")

    receiver_task = asyncio.create_task(audio_receiver())

    try:
        # 1. Greet the user
        initial_greeting = (f"Hi, am I speaking with {lead_name}?" if lead_name
                            else f"Hi, thank you for calling {config.YOUR_BUSINESS_NAME}. My name is Sky, how can I help?")
        greet_event = asyncio.Event()
        await generate_and_stream_audio(initial_greeting, websocket, stream_sid, greet_event)

        # 2. Main conversation loop
        while not stop_event.is_set():
            streaming_config = get_stt_config()
            
            async def audio_generator():
                while not stop_event.is_set():
                    chunk = await audio_queue.get()
                    if chunk is None: break
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            try:
                # This call will block until a final transcript is received
                responses = await asyncio.to_thread(
                    speech_client.streaming_recognize,
                    config=streaming_config,
                    requests=audio_generator()
                )

                transcript = ""
                for response in responses:
                    if response.results and response.results[0].alternatives:
                        transcript = response.results[0].alternatives[0].transcript.strip()
                        break # We only need the first final result due to single_utterance

                if transcript:
                    logging.info(f"User (call {call_sid}): {transcript}")

                    # If an agent is currently speaking, interrupt it
                    if active_agent_task and not active_agent_task.done() and interruption_event:
                        logging.info(f"User interruption detected for call {call_sid}. Cancelling previous agent response.")
                        interruption_event.set()
                        await asyncio.sleep(0.1) # Give a moment for the task to stop streaming

                    # Start processing the new user input
                    active_agent_task, interruption_event = await handle_agent_response(transcript, call_sid, stream_sid, websocket)

            except Exception as e:
                # Handle stream timeouts or other STT errors
                if "Deadline Exceeded" in str(e):
                    logging.warning(f"STT stream timed out for {call_sid}. Listening again.")
                else:
                    logging.error(f"STT recognizer error for {call_sid}: {e}", exc_info=True)
                # If there's an error, break the loop to be safe
                break

    except Exception as e:
        logging.error(f"Error in transcription_agent_task for {call_sid}: {e}", exc_info=True)
    finally:
        stop_event.set()
        receiver_task.cancel()
        if active_agent_task:
            active_agent_task.cancel()
        
        if call_sid in conversation_history:
            del conversation_history[call_sid]
            
        logging.info(f"Agent task finished for call {call_sid}.")


# --- FastAPI Endpoints ---
@app.post("/inbound_call")
async def handle_inbound_call():
    logging.info("Inbound call received")
    response = VoiceResponse()
    connect = Connect()
    # Ensure the websocket URL is correct for your deployment
    connect.stream(url=f"wss://realestate-voiceai-receptionist.onrender.com/ws")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str | None = None):
    await websocket.accept()
    logging.info(f"WebSocket connection accepted. Lead name: {name if name else 'Inbound'}")
    call_sid, stream_sid = None, None
    try:
        # Wait for the start event from Twilio
        message = await asyncio.wait_for(websocket.receive_text(), timeout=5)
        data = json.loads(message)
        if data.get("event") == "start":
            call_sid = data['start']['callSid']
            stream_sid = data['start']['streamSid']
            logging.info(f"WebSocket received start event for call {call_sid}, stream {stream_sid}")
            
            # Start the main agent task
            await transcription_agent_task(websocket, call_sid, stream_sid, lead_name=name)
        else:
            logging.error(f"Received unexpected event '{data.get('event')}' before 'start'.")

    except asyncio.TimeoutError:
        logging.error("WebSocket connection timed out before 'start' event.")
    except WebSocketDisconnect:
        logging.warning(f"WebSocket client disconnected for call SID: {call_sid or 'Unknown'}.")
    except Exception as e:
        logging.error(f"Error in WebSocket endpoint for call SID {call_sid or 'Unknown'}: {e}", exc_info=True)
    finally:
        logging.info(f"WebSocket endpoint closing for call SID: {call_sid or 'Unknown'}.")

@app.post("/start_outbound_campaign")
async def start_outbound_campaign(file: UploadFile = File(...)):
    global campaign_in_progress
    if campaign_in_progress:
        return {"status": "error", "message": "A campaign is already in progress."}
    logging.info("Received request to start outbound campaign.")
    try:
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
