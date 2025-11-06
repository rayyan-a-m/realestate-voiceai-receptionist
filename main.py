import asyncio
import base64
import json
import csv
import io
import logging
from urllib.parse import urlparse

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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

import config
from prompts import prompt
from google_calendar import find_available_slots, book_appointment

# --- Initialization ---
app = FastAPI()

# Add CORS middleware to allow all origins
# This is crucial for services like Twilio to connect via WebSocket
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

twilio_client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
deepgram_client = DeepgramClient(config.DEEPGRAM_API_KEY)
elevenlabs_client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)

logging.info("Service initialized")

# --- LangChain Agent Setup ---
tools = [find_available_slots, book_appointment]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# Bind tools directly to the LLM for tool calling
llm_with_tools = llm.bind_tools(tools)

# Create a simple agent executor function
async def run_agent(user_input: str, chat_history: list):
    """Run the agent with tool calling capability."""
    logging.info(f"Running agent with user_input: {user_input}")
    # Format the prompt with history
    messages = [
        {"role": "system", "content": prompt.messages[0].prompt.template}
    ]
    
    # Add chat history
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})
    
    # Invoke LLM with tools
    response = await llm_with_tools.ainvoke(messages)
    
    # Check if tool calls are needed
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logging.info(f"LLM requested tool calls: {response.tool_calls}")
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Execute the tool
            logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
            if tool_name == 'find_available_slots':
                result = find_available_slots.invoke(tool_args)
            elif tool_name == 'book_appointment':
                result = book_appointment.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            tool_results.append(str(result))
        
        # Add tool results to messages and get final response
        messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
        messages.append({"role": "tool", "content": "\n".join(tool_results)})
        final_response = await llm_with_tools.ainvoke(messages)
        logging.info(f"Agent response after tool call: {final_response.content}")
        return final_response.content
    
    logging.info(f"Agent response: {response.content}")
    return response.content

# --- NEW: Outbound Campaign Management ---
outbound_leads_queue = asyncio.Queue()
campaign_in_progress = False

# This is the worker that processes the queue
async def campaign_worker():
    global campaign_in_progress
    campaign_in_progress = True
    logging.info("Starting outbound campaign worker...")

    while not outbound_leads_queue.empty():
        lead = await outbound_leads_queue.get()
        logging.info(f"Processing lead: {lead['first_name']} {lead['last_name']} at {lead['phone']}")
        
        try:
            # For outbound calls, Twilio needs TwiML instructions.
            # We instruct Twilio to call the number and then connect to our WebSocket stream.
            # The `url` in `<Stream>` must be a `wss` URL.
            host = urlparse(config.AGENT_HOST_URL).netloc
            logging.info(f"Parsed host for WebSocket URL: {host}")
            websocket_url = f"wss://realestate-voiceai-receptionist.onrender.com/ws?name={lead['first_name']}"
            
            # We now use TwiML to direct the call
            twiml_response = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            twiml_response.append(connect)
            
            logging.info(f"Initiating outbound call to {lead['phone']} with TwiML: {str(twiml_response)}")

            call = twilio_client.calls.create(
                to=lead['phone'],
                from_=config.TWILIO_PHONE_NUMBER,
                twiml=str(twiml_response)
            )
            logging.info(f"Outbound call initiated to {lead['phone']}, SID: {call.sid}")
            
            # Wait between calls to not be overwhelming
            await asyncio.sleep(15) 

        except Exception as e:
            logging.error(f"Failed to call lead {lead['first_name']}: {e}", exc_info=True)
        
        outbound_leads_queue.task_done()
    
    logging.info("Outbound campaign finished.")
    campaign_in_progress = False

# --- Real-time Transcription & Agent Logic ---
class ConnectionManager:
    """Manages WebSocket connections for real-time transcription."""
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, call_sid: str, websocket: WebSocket):
        self.active_connections[call_sid] = websocket
        logging.info(f"WebSocket registered for call SID: {call_sid}")

    def disconnect(self, call_sid: str):
        if call_sid in self.active_connections:
            del self.active_connections[call_sid]
            logging.info(f"WebSocket disconnected for call SID: {call_sid}")

manager = ConnectionManager()
conversation_history = {}

async def transcription_agent_task(websocket: WebSocket, call_sid: str, stream_sid: str, lead_name: str | None = None):
    """The main task that handles real-time transcription and agent responses."""
    logging.info(f"Starting transcription agent for call {call_sid}, stream {stream_sid}")
    deepgram_conn = None
    try:
        deepgram_conn = deepgram_client.listen.asynclive.v("1")
        
        async def on_message(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript and result.is_final:
                logging.info(f"User (call {call_sid}): {transcript}")
                
                chat_history = conversation_history.get(call_sid, [])
                
                # Invoke the agent
                output_text = await run_agent(transcript, chat_history)
                logging.info(f"Agent (call {call_sid}): {output_text}")
                
                # Save history
                chat_history.append(HumanMessage(content=transcript))
                chat_history.append(AIMessage(content=output_text))
                conversation_history[call_sid] = chat_history
                
                # Generate audio and stream back to Twilio
                await generate_and_stream_audio(output_text, websocket, stream_sid)

        deepgram_conn.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="mulaw",
            channels=1,
            sample_rate=8000,
            endpointing=300, # Milliseconds of silence to consider an utterance complete
        )
        await deepgram_conn.start(options)
        
        initial_greeting = f"Hi, am I speaking with {lead_name}?" if lead_name else f"Thank you for calling {config.YOUR_BUSINESS_NAME}, my name is Sky. How can I help you today?"
        logging.info(f"Initial greeting for call {call_sid}: {initial_greeting}")
        await generate_and_stream_audio(initial_greeting, websocket, stream_sid)

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            if data["event"] == "media":
                payload = base64.b64decode(data["media"]["payload"])
                await deepgram_conn.send(payload)

    except WebSocketDisconnect:
        logging.warning(f"WebSocket disconnected for call {call_sid}")
    except Exception as e:
        logging.error(f"Error in transcription_agent_task for {call_sid}: {e}", exc_info=True)
    finally:
        if deepgram_conn:
            await deepgram_conn.finish()
        manager.disconnect(call_sid)
        if call_sid in conversation_history:
            del conversation_history[call_sid]
        logging.info(f"Connection for call {call_sid} closed.")


async def generate_and_stream_audio(text: str, websocket: WebSocket, stream_sid: str):
    """Generates audio using ElevenLabs and streams it to Twilio via WebSocket."""
    logging.info(f"Generating audio for stream {stream_sid}: '{text}'")
    try:
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=text,
            voice=config.ELEVENLABS_VOICE_ID,
            model="eleven_turbo_v2",
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format="mulaw_8000"
        )

        async for chunk in audio_stream:
            if chunk:
                encoded_chunk = base64.b64encode(chunk).decode('utf-8')
                media_message = {
                    "event": "media",
                    "streamSid": stream_sid, 
                    "media": {"payload": encoded_chunk}
                }
                await websocket.send_text(json.dumps(media_message))

        # Send a "mark" message to indicate the end of the audio stream
        mark_message = {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": "end_of_turn"}
        }
        await websocket.send_text(json.dumps(mark_message))
        logging.info(f"Finished streaming audio for stream {stream_sid}")
    except Exception as e:
        logging.error(f"Error generating or streaming audio for stream {stream_sid}: {e}", exc_info=True)


# --- FastAPI Endpoints ---

@app.post("/inbound_call")
async def handle_inbound_call():
    """Handles incoming calls from Twilio."""
    logging.info("Inbound call received")
    response = VoiceResponse()
    connect = Connect()
    host = urlparse(config.AGENT_HOST_URL).netloc
    connect.stream(url=f"wss://realestate-voiceai-receptionist.onrender.com/ws")
    response.append(connect)
    logging.info(f"Responding to inbound call with TwiML: {str(response)}")
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str | None = None):
    """The main WebSocket endpoint for Twilio media streams."""
    # The headers that FastAPI/Starlette use to check for origins are not
    # correctly forwarded by some load balancers/proxies, including Render.
    # We can manually set the `host` to ensure the check passes.
    # This is a workaround for the "403 Forbidden" error.
    host = urlparse(config.AGENT_HOST_URL).netloc
    websocket.scope['headers'] = [
        (b'host', b'realestate-voiceai-receptionist.onrender.com') 
        if item[0] == b'host' else item 
        for item in websocket.scope['headers']
    ]
    
    await websocket.accept()
    logging.info(f"WebSocket connection accepted. Lead name: {name if name else 'Inbound'}")
    
    call_sid = None
    stream_sid = None
    
    try:
        # Twilio sends a 'connected' message before the 'start' message.
        # We need to loop until we get the 'start' message.
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "connected":
                logging.info("Received 'connected' event from Twilio.")
                continue
            
            if event == "start":
                call_sid = data['start']['callSid']
                stream_sid = data['start']['streamSid']
                logging.info(f"WebSocket received start event for call {call_sid}, stream {stream_sid}")
                break # Exit the loop once we have the start event
            
            # If we receive something else before 'start', it's unexpected.
            logging.error(f"Received unexpected event '{event}' before 'start'.")
            await websocket.close(code=1002) # Protocol error
            return

        # Register the connection now that we have the call_sid
        await manager.connect(call_sid, websocket)
        
        # Start the agent task to handle the call
        agent_task = asyncio.create_task(
            transcription_agent_task(websocket, call_sid, stream_sid, lead_name=name)
        )

        # Keep the connection alive by listening for media and other messages.
        # The agent task will handle the media messages.
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        logging.warning(f"WebSocket client disconnected for call SID: {call_sid if call_sid else 'Unknown'}.")
    except Exception as e:
        logging.error(f"Error in WebSocket endpoint for call SID {call_sid if call_sid else 'Unknown'}: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens if the connection is closed for any reason
        if call_sid:
            manager.disconnect(call_sid)
        logging.info(f"WebSocket endpoint closing for call SID: {call_sid if call_sid else 'Unknown'}.")

@app.post("/start_outbound_campaign")
async def start_outbound_campaign(file: UploadFile = File(...)):
    """
    Starts an outbound calling campaign from a CSV file.
    The CSV should have 'first_name', 'last_name', and 'phone' headers.
    """
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
            # Start the background worker
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
