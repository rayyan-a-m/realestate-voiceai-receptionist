import asyncio
import logging
import json
import sys
import os
from queue import Queue
import numpy as np
import sounddevice as sd
import contextlib
import aiohttp
import ssl
import certifi
import wave

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Imports ---
try:
    from elevenlabs.client import AsyncElevenLabs
    from elevenlabs import VoiceSettings
except ImportError as e:
    logging.error(f"Failed to import a dependency: {e}")
    logging.error("Please ensure your virtual environment is active and you have run 'python -m pip install -r requirements.txt'")
    exit(1)

from langchain_core.messages import HumanMessage, AIMessage
import config
from main import run_agent
 

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 1

# --- Initialization ---
elevenlabs_client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)
audio_queue = Queue()
chat_history = []
is_speaking = False  # specifically while TTS audio is playing
agent_busy = False   # covers the full turn: LLM thinking + TTS playback

def audio_callback(indata, frames, time, status):
    """This is called by the sounddevice stream for each audio block."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    # Only queue audio if the agent is not busy (not thinking or speaking)
    if not agent_busy and not is_speaking:
        audio_queue.put(bytes(indata))

def _to_text(content) -> str:
    """Normalize agent/LLM content to plain string for TTS (handles list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Handle LangChain AIMessage/HumanMessage objects
    try:
        if hasattr(content, "content"):
            return _to_text(content.content)
    except Exception:
        pass
    # Handle list-of-parts (Gemini style)
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
    # Fallback: try common dict shapes
    if isinstance(content, dict):
        if 'text' in content:
            return str(content['text'])
        if 'content' in content:
            return _to_text(content['content'])
    return str(content) if content is not None else ""

def _clear_audio_queue():
    """Remove any pending audio frames from the queue to avoid stale/echo input between turns."""
    try:
        while True:
            audio_queue.get_nowait()
    except Exception:
        pass

async def handle_agent_response(transcript):
    """Gets response from LLM, synthesizes audio, and plays it."""
    global is_speaking, agent_busy
    
    # Mark agent busy for the entire turn and drop any queued frames
    agent_busy = True
    _clear_audio_queue()
    logging.info("Agent turn started (busy).")

    agent_response = await run_agent(transcript, chat_history)
    tts_text = _to_text(agent_response)
    logging.info(f"Agent says: {tts_text}")
    chat_history.append(HumanMessage(content=transcript))
    chat_history.append(AIMessage(content=tts_text))
    
    try:
        # Skip if empty
        if not tts_text or not tts_text.strip():
            logging.warning("TTS text is empty; skipping audio generation.")
            agent_busy = False
            return

        # Enter speaking mode early and drop any queued mic frames to prevent echo/stale audio
        is_speaking = True
        _clear_audio_queue()
        logging.info("Generating audio response from ElevenLabs...")

        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=tts_text,
            voice_id=config.ELEVENLABS_VOICE_ID,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format='pcm_16000'
        )

        # Collect streamed PCM bytes
        audio_chunks = []
        async for chunk in audio_stream:
            if chunk:
                audio_chunks.append(chunk)

        if audio_chunks:
            pcm_bytes = b"".join(audio_chunks)
            # Debug: save to WAV
            try:
                with wave.open("last_tts.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm_bytes)
                logging.info("Wrote ElevenLabs audio to last_tts.wav for debugging.")
            except Exception as we:
                logging.debug(f"Failed to write debug WAV: {we}")

            # Convert to numpy int16
            samples_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            # Compute RMS to detect silence
            if samples_i16.size == 0:
                logging.warning("Received empty audio from ElevenLabs; skipping playback.")
                is_speaking = False
                return
            rms = float(np.sqrt(np.mean((samples_i16.astype(np.float32))**2)))
            peak = int(np.max(np.abs(samples_i16)))
            logging.info(f"Audio stats — frames: {len(samples_i16)}, RMS: {rms:.2f}, peak: {peak}")

            # Convert to float32 in [-1, 1] for safer playback
            samples_f32 = (samples_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

            # Optional gain if too quiet
            if rms < 500.0:
                gain = 1.8
                samples_f32 = np.clip(samples_f32 * gain, -1.0, 1.0)
                logging.info(f"Applied gain x{gain} to boost quiet audio.")

            # Stop any current playback and play
            sd.stop()
            logging.info("Playing audio response via speakers...")
            sd.play(samples_f32, samplerate=SAMPLE_RATE, blocking=False)
            sd.wait()
            # small cooldown to avoid capturing trailing playback residuals
            await asyncio.sleep(0.25)
            # Clear any residual frames captured right after playback before resuming listen
            _clear_audio_queue()
            is_speaking = False  # Resume listening
            agent_busy = False
            logging.info("Agent turn finished. Listening resumed.")

    except Exception as e:
        logging.error(f"ElevenLabs error: {e}")
        is_speaking = False
        agent_busy = False  # Ensure flags are reset on error

async def microphone_sender(ws):
    """Streams audio from the microphone to the Deepgram websocket (aiohttp)."""
    while True:
        try:
            if ws.closed:
                break
            data = audio_queue.get()
            await ws.send_bytes(data)
        except ConnectionResetError:
            logging.error("WebSocket transport reset while sending audio. Stopping sender.")
            break
        except asyncio.CancelledError:
            break

async def deepgram_receiver(ws, ready_event: asyncio.Event):
    """Receives transcripts from Deepgram websocket and triggers the agent on final results (aiohttp)."""
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    resp = json.loads(msg.data)
                except Exception:
                    continue
                # Mark ready once first text message is received from server
                if not ready_event.is_set():
                    ready_event.set()
                # Deepgram realtime messages include is_final and transcript
                is_final = resp.get("is_final") or resp.get("speech_final")
                channel = resp.get("channel") or {}
                alts = channel.get("alternatives") or []
                transcript = alts[0].get("transcript") if alts else ""
                if is_final and transcript and transcript.strip() and not agent_busy and not is_speaking:
                    logging.info(f"You said: {transcript}")
                    asyncio.create_task(handle_agent_response(transcript))
            elif msg.type == aiohttp.WSMsgType.BINARY:
                # Some servers may send binary control frames; consider this as ready
                if not ready_event.is_set():
                    ready_event.set()
                continue
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logging.error(f"WebSocket error: {ws.exception()}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                logging.error("WebSocket closed by server.")
                break
            # Binary messages from Deepgram aren't expected for transcripts; ignore
    except Exception as e:
        logging.error(f"Receiver error: {e}")

async def main():
    """Main function to run the local voice agent test (no Twilio)."""
    try:
        # Log versions to help diagnose environment issues
        try:
            import websockets as _ws
            logging.info(f"Python {sys.version.split()[0]} | websockets={getattr(_ws, '__version__', 'unknown')}")
        except Exception:
            logging.info(f"Python {sys.version.split()[0]} | websockets=unavailable for import")

        # Log audio devices and optionally set output device from env
        try:
            devices = sd.query_devices()
            default_in, default_out = sd.default.device
            logging.info(f"Default audio devices — input: {default_in}, output: {default_out}")
            # Short device summary
            output_devices = [f"{i}: {d['name']} ({d['max_output_channels']}ch)" for i, d in enumerate(devices) if d.get('max_output_channels', 0) > 0]
            logging.info("Output devices: " + "; ".join(output_devices))
            # Allow choosing device via env vars
            env_out = os.getenv("SD_OUTPUT_DEVICE") or os.getenv("OUTPUT_DEVICE")
            if env_out is not None:
                try:
                    # Try numeric index; fallback to name
                    out_dev = int(env_out) if env_out.isdigit() else env_out
                    sd.default.device = (sd.default.device[0], out_dev)
                    logging.info(f"Using output device from env: {out_dev}")
                except Exception as de:
                    logging.warning(f"Failed to set output device '{env_out}': {de}")
        except Exception as dev_err:
            logging.debug(f"Audio device query failed: {dev_err}")

        # Build Deepgram realtime URL and headers
        url = (
            "wss://api.deepgram.com/v1/listen?"
            f"model=nova-2&language=en-US&smart_format=true&encoding=linear16&sample_rate={SAMPLE_RATE}&channels={CHANNELS}&endpointing=500"
        )
        # Connect to Deepgram websocket using aiohttp
        # Use certifi CA bundle to avoid macOS local issuer issues
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}, connector=connector) as session:
            async with session.ws_connect(url, heartbeat=20) as ws:
                logging.info("Connected to Deepgram. Local test starting...")

                # Start microphone input
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=audio_callback):
                    logging.info("Speak into your microphone. The agent will respond when you pause.")

                    # Start receiver first and wait for server message before sending audio
                    ready_event = asyncio.Event()
                    receiver_task = asyncio.create_task(deepgram_receiver(ws, ready_event))
                    try:
                        await asyncio.wait_for(ready_event.wait(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logging.warning("No server message within 10s; starting audio stream anyway.")

                    sender_task = asyncio.create_task(microphone_sender(ws))

                    # Keep process alive until interrupted
                    try:
                        await asyncio.gather(sender_task, receiver_task)
                    except asyncio.CancelledError:
                        pass
                    finally:
                        for t in (sender_task, receiver_task):
                            t.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(sender_task, receiver_task)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logging.info("Test finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping test...")
