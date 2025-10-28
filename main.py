import asyncio
import base64
import json
import csv
import io
import pytz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
import uvicorn
import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage

from deepgram import DeepgramClient, LiveTranscriptionEvents
from deepgram.options import LiveOptions
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings

import config
from prompts import prompt
from google_calendar import find_available_slots, book_appointment

# --- Initialization ---
app = FastAPI()
twilio_client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
deepgram_client = DeepgramClient(config.DEEPGRAM_API_KEY)
elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

# --- LangChain Agent Setup ---
tools = [find_available_slots, book_appointment]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- NEW: Outbound Campaign Management ---
outbound_leads_queue = asyncio.Queue()
campaign_in_progress = False

# This is the worker that processes the queue
async def campaign_worker():
    global campaign_in_progress
    campaign_in_progress = True
    print("Starting outbound campaign worker...")

    while not outbound_leads_queue.empty():
        lead = await outbound_leads_queue.get()
        print(f"Processing lead: {lead['first_name']} {lead['last_name']} at {lead['phone']}")
        
        try:
            # We add lead info to the TwiML URL to pass context to the WebSocket
            twiml_response = f'<Response><Connect><Stream url="wss://{config.AGENT_HOST_URL.split("//")[1]}/ws?name={lead["first_name"]}" /></Connect></Response>'
            
            call = twilio_client.calls.create(
                to=lead['phone'],
                from_=config.TWILIO_PHONE_NUMBER,
                twiml=twiml_response
            )
            print(f"Outbound call initiated to {lead['phone']}, SID: {call.sid}")
            
            # Wait between calls to not be overwhelming
            await asyncio.sleep(15) 

        except Exception as e:
            print(f"Failed to call lead {lead['first_name']}: {e}")
        
        outbound_leads_queue.task_done()
    
    print("Outbound campaign finished.")
    campaign_in_progress = False

# --- Real-time Transcription & Agent Logic ---
class ConnectionManager:
    """Manages WebSocket connections for real-time transcription."""
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, call_sid: str):
        await websocket.accept()
        self.active_connections[call_sid] = websocket

    def disconnect(self, call_sid: str):
        if call_sid in self.active_connections:
            del self.active_connections[call_sid]

manager = ConnectionManager()
conversation_history = {}

async def transcription_agent_task(websocket: WebSocket, call_sid: str, stream_sid: str, lead_name: str | None = None):
    """The main task that handles real-time transcription and agent responses."""
    try:
        deepgram_conn = await deepgram_client.listen.asynclive.v("1")
        
        initial_greeting = f"Hi, am I speaking with {lead_name}?" if lead_name else f"Thank you for calling {config.YOUR_BUSINESS_NAME}, my name is Sky. How can I help you today?"
        await generate_and_stream_audio(initial_greeting, websocket, stream_sid)

        async def on_message(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript and result.is_final:
                print(f"User: {transcript}")
                
                chat_history = conversation_history.get(call_sid, [])
                
                # Invoke the agent
                response = await agent_executor.ainvoke({
                    "input": transcript,
                    "chat_history": chat_history,
                })
                output_text = response["output"]
                print(f"Agent: {output_text}")
                
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

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            if data["event"] == "media":
                payload = base64.b64decode(data["media"]["payload"])
                await deepgram_conn.send(payload)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for call {call_sid}")
    except Exception as e:
        print(f"Error in transcription_agent_task for {call_sid}: {e}")
    finally:
        manager.disconnect(call_sid)
        if call_sid in conversation_history:
            del conversation_history[call_sid]
        print(f"Connection for call {call_sid} closed.")


async def generate_and_stream_audio(text: str, websocket: WebSocket, stream_sid: str):
    """Generates audio using ElevenLabs and streams it to Twilio via WebSocket."""
    audio_stream = await elevenlabs_client.generate(
        text=text,
        voice=config.ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2",
        stream=True,
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


# --- FastAPI Endpoints ---

@app.post("/inbound_call")
async def handle_inbound_call():
    """Handles incoming calls from Twilio."""
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{config.AGENT_HOST_URL.split('//')[1]}/ws")
    response.append(connect)
    print("Inbound call received, connecting to WebSocket.")
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str | None = None):
    """The main WebSocket endpoint for Twilio media streams."""
    try:
        message = await websocket.receive_text()
        data = json.loads(message)
        if data['event'] != 'start':
            return

        call_sid = data['start']['callSid']
        stream_sid = data['start']['streamSid']
        await manager.connect(websocket, call_sid)
        
        await transcription_agent_task(websocket, call_sid, stream_sid, name)

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.post("/start_outbound_campaign")
async def start_outbound_campaign(file: UploadFile = File(...)):
    """
    Upload a CSV with 'first_name', 'last_name', 'phone' columns to start a campaign.
    """
    global campaign_in_progress
    if campaign_in_progress:
        return {"status": "error", "message": "A campaign is already in progress."}

    contents = await file.read()
    buffer = io.StringIO(contents.decode('utf-8'))
    csv_reader = csv.DictReader(buffer)
    
    for row in csv_reader:
        await outbound_leads_queue.put(row)
    
    if outbound_leads_queue.empty():
        return {"status": "error", "message": "CSV file is empty or has invalid format."}

    # Start the worker in the background
    asyncio.create_task(campaign_worker())

    return {"status": "success", "message": f"Campaign started with {outbound_leads_queue.qsize()} leads."}


if __name__ == "__main__":
    print(f"Starting server. Make sure your AGENT_HOST_URL is set to a public URL.")
    if config.AGENT_HOST_URL:
        print(f"Twilio will connect to: wss://{config.AGENT_HOST_URL.split('//')[1]}/ws")
    else:
        print("AGENT_HOST_URL is not set. Twilio connection will fail.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
