import asyncio
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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

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
elevenlabs_client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)

logging.info("Service initialized")

# --- LangChain Agent Setup ---
tools = [find_available_slots, book_appointment]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=config.GOOGLE_API_KEY)  # Use a currently available model per ListModels; Flash is faster for voice interactions
llm_with_tools = llm.bind_tools(tools)

def _to_text(content) -> str:
    """Normalize LLM content to a plain string for TTS.
    Handles strings and lists like [{'type': 'text', 'text': '...'}].
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get('text') or ''
                if txt:
                    parts.append(txt)
            else:
                parts.append(str(part))
        return " ".join(p.strip() for p in parts if p and p.strip())
    if isinstance(content, dict):
        if 'text' in content:
            return str(content['text'])
        if 'content' in content:
            return _to_text(content['content'])
    # Fallback for LC message objects
    if hasattr(content, 'content'):
        return _to_text(getattr(content, 'content'))
    return str(content)

async def run_agent(user_input: str, chat_history: list):
    """Run the LLM with optional tool-calling and return plain text for TTS."""
    logging.info(f"Running agent with user_input: {user_input}")

    # Build LC message list
    lc_messages = [SystemMessage(content=prompt.messages[0].prompt.template)]
    lc_messages.extend(chat_history)
    lc_messages.append(HumanMessage(content=user_input))

    try:
        logging.info("Invoking LLM...")
        response = await asyncio.wait_for(llm_with_tools.ainvoke(lc_messages), timeout=20)
        logging.info(f"LLM response received: {response.content}")
    except asyncio.TimeoutError:
        logging.error("LLM timed out after 20s. Returning a brief fallback reply.")
        return "I'm here. How can I help you today?"
    except Exception as e:
        logging.error(f"LLM error: {e}", exc_info=True)
        return "Sorry, I had trouble responding. Please try again."

    # Handle potential tool calls
    tool_calls = getattr(response, 'tool_calls', None) or getattr(response, 'additional_kwargs', {}).get('tool_calls')
    if tool_calls:
        logging.info(f"LLM requested tool calls: {tool_calls}")
        tool_results = []
        for call in tool_calls:
            try:
                tool_name = call.get('name') if isinstance(call, dict) else call.name
                tool_args = call.get('args') if isinstance(call, dict) else call.args
                logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                if tool_name == 'find_available_slots':
                    result = find_available_slots.invoke(tool_args or {})
                elif tool_name == 'book_appointment':
                    result = book_appointment.invoke(tool_args or {})
                else:
                    result = f"Unknown tool: {tool_name}"
            except Exception as e:
                logging.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
                result = f"Tool '{tool_name}' errored."
            tool_results.append(str(result))

        # Ask model to finalize with tool results
        lc_messages.append(AIMessage(content=_to_text(response.content)))
        lc_messages.append(HumanMessage(content=f"Tool results:\n{os.linesep.join(tool_results)}\nPlease respond to the user accordingly."))

        try:
            final_response = await asyncio.wait_for(llm_with_tools.ainvoke(lc_messages), timeout=20)
            return _to_text(final_response.content)
        except asyncio.TimeoutError:
            logging.error("LLM timed out after tools. Returning brief confirmation.")
            return "I've checked the calendar and can help you book. What time works for you?"
        except Exception as e:
            logging.error(f"LLM error after tools: {e}", exc_info=True)
            return "I ran into an error finalizing that. Could you rephrase your request?"

    # No tools – return normalized text
    return _to_text(response.content)

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
            websocket_url = f"wss://realestate-voiceai-receptionist.onrender.com/ws?name={lead['first_name']}"
            twiml_response = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            twiml_response.append(connect)
            logging.info(f"Initiating outbound call to {lead['phone']} with TwiML: {str(twiml_response)}")
            call = twilio_client.calls.create(to=lead['phone'], from_=config.TWILIO_PHONE_NUMBER, twiml=str(twiml_response))
            logging.info(f"Outbound call initiated to {lead['phone']}, SID: {call.sid}")
            await asyncio.sleep(15)
        except Exception as e:
            logging.error(f"Failed to call lead {lead['first_name']}: {e}", exc_info=True)
        outbound_leads_queue.task_done()
    logging.info("Outbound campaign finished.")
    campaign_in_progress = False

# --- Real-time Transcription & Agent Logic ---
conversation_history = {}
DEEPGRAM_URL = (
    f"wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&smart_format=true"
    f"&encoding=mulaw&sample_rate=8000&channels=1&endpointing=300"
)

async def transcription_agent_task(twilio_websocket: WebSocket, call_sid: str, stream_sid: str, lead_name: str | None = None):
    """Handles the full lifecycle of a voice call, including transcription and agent interaction."""
    logging.info(f"Starting agent task for call {call_sid}, stream {stream_sid}")
    headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}

    async with websockets.connect(DEEPGRAM_URL, extra_headers=headers) as deepgram_ws:
        logging.info(f"Connected to Deepgram for call {call_sid}")

        async def twilio_receiver(twilio_ws, deepgram_ws):
            """Receives audio from Twilio and forwards it to Deepgram."""
            try:
                while True:
                    message_str = await twilio_ws.receive_text()
                    data = json.loads(message_str)
                    if data.get("event") == "media":
                        payload = base64.b64decode(data["media"]["payload"])
                        await deepgram_ws.send(payload)
                    elif data.get("event") == "stop":
                        logging.info(f"Twilio stop event received for call {call_sid}")
                        break
            except WebSocketDisconnect:
                logging.warning(f"Twilio WebSocket disconnected for call {call_sid}")
            except Exception as e:
                logging.error(f"Error in twilio_receiver for {call_sid}: {e}")

        async def deepgram_receiver(deepgram_ws, twilio_ws):
            """Receives transcripts from Deepgram and triggers the agent."""
            try:
                async for msg in deepgram_ws:
                    resp = json.loads(msg)
                    if resp.get("type") == "SpeechFinal" or (resp.get("is_final") and resp.get("speech_final")):
                        transcript = resp["channel"]["alternatives"][0]["transcript"]
                        if transcript.strip():
                            logging.info(f"User (call {call_sid}): {transcript}")

                            chat_history = conversation_history.get(call_sid, [])
                            agent_response = await run_agent(transcript, chat_history)
                            logging.info(f"Agent (call {call_sid}): {agent_response}")

                            chat_history.append(HumanMessage(content=transcript))
                            chat_history.append(AIMessage(content=agent_response))
                            conversation_history[call_sid] = chat_history

                            await generate_and_stream_audio(agent_response, twilio_ws, stream_sid)
            except Exception as e:
                logging.error(f"Error in deepgram_receiver for {call_sid}: {e}")

        try:
            initial_greeting = (
                f"Hi, am I speaking with {lead_name}?" if lead_name else
                f"Thank you for calling {config.YOUR_BUSINESS_NAME}, my name is Sky. How can I help you today?"
            )
            logging.info(f"Initial greeting for call {call_sid}: {initial_greeting}")
            await generate_and_stream_audio(initial_greeting, twilio_websocket, stream_sid)

            twilio_task = asyncio.create_task(twilio_receiver(twilio_websocket, deepgram_ws))
            deepgram_task = asyncio.create_task(deepgram_receiver(deepgram_ws, twilio_websocket))
            await asyncio.gather(twilio_task, deepgram_task)
        except WebSocketDisconnect:
            logging.warning(f"WebSocket disconnected during agent task for call {call_sid}")
        except Exception as e:
            logging.error(f"Error in transcription_agent_task for {call_sid}: {e}", exc_info=True)
        finally:
            if call_sid in conversation_history:
                del conversation_history[call_sid]
            logging.info(f"Agent task finished for call {call_sid}.")

async def generate_and_stream_audio(text: str, websocket: WebSocket, stream_sid: str):
    logging.info(f"Generating audio for stream {stream_sid}: '{text}'")
    try:
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=text,
            voice_id=config.ELEVENLABS_VOICE_ID,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format="ulaw_8000"
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
    logging.info("Inbound call received")
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://realestate-voiceai-receptionist.onrender.com/ws")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str | None = None):
    await websocket.accept()
    logging.info(f"WebSocket connection accepted. Lead name: {name if name else 'Inbound'}")
    call_sid, stream_sid = None, None
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")
            if event == "connected":
                continue
            if event == "start":
                call_sid = data['start']['callSid']
                stream_sid = data['start']['streamSid']
                logging.info(f"WebSocket received start event for call {call_sid}, stream {stream_sid}")
                break
            logging.error(f"Received unexpected event '{event}' before 'start'.")
            return
        
        await transcription_agent_task(websocket, call_sid, stream_sid, lead_name=name)

    except WebSocketDisconnect:
        logging.warning(f"WebSocket client disconnected for call SID: {call_sid if call_sid else 'Unknown'}.")
    except Exception as e:
        logging.error(f"Error in WebSocket endpoint for call SID {call_sid if call_sid else 'Unknown'}: {e}", exc_info=True)
    finally:
        logging.info(f"WebSocket endpoint closing for call SID: {call_sid if call_sid else 'Unknown'}.")

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
