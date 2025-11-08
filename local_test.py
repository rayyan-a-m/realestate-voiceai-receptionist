import time
import asyncio
import logging
import os
import sys
import time
from queue import Queue
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from vertexai import agent_engines
import webrtcvad
import threading

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Imports ---
try:
    from google.cloud import speech_v1 as speech
    from google.cloud import texttospeech_v1 as texttospeech
except ImportError as e:
    logging.error(f"Failed to import Google Cloud libraries: {e}")
    logging.error("Please ensure your virtual environment is active and you have run 'pip install -r requirements.txt'")
    sys.exit(1)

from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
SAMPLE_RATE = 16000  # Required for Google STT
CHANNELS = 1
BLOCK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms of audio
VAD_FRAME_DURATION_MS = 30  # VAD frame duration must be 10, 20, or 30 ms
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0))
VAD_BYTES_PER_FRAME = VAD_FRAME_SAMPLES * 2 # 16-bit audio (2 bytes/sample)

# --- Initialization ---
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
audio_queue = Queue()
chat_history = []
is_speaking = False
agent_busy = False
interruption_event = threading.Event() # Use threading.Event for cross-thread safety
vad = webrtcvad.Vad(1)  # Set VAD aggressiveness from 0 (least aggressive) to 3 (most aggressive)
vad_buffer = bytearray()
current_playback_thread = None

# --- LLM Agent Initialization ---
try:
    agent = agent_engines.LangchainAgent(
        model="gemini-2.5-flash-lite",
        tools=[],  # No tools needed for local voice test
        model_kwargs={
            "temperature": 0.0,
            "max_output_tokens": 256,
            "top_p": 0.95,
        },
    )
    logging.info("LLM Agent initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize LLM Agent: {e}", exc_info=True)
    sys.exit(1)

# --- Audio Playback Thread ---
class PlaybackThread(threading.Thread):
    def __init__(self, audio_content, samplerate, interruption_event):
        super().__init__()
        self.audio_content = audio_content
        self.samplerate = samplerate
        self.interruption_event = interruption_event
        self.playback_finished = threading.Event()

    def run(self):
        global is_speaking
        is_speaking = True
        logging.info("Playing audio response...")
        print("--- AGENT SPEAKING ---")
        
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype='int16',
                finished_callback=self.playback_finished.set
            )
            stream.start()
            # Write the audio data. This is non-blocking.
            stream.write(self.audio_content)

            # Wait until playback is finished or an interruption occurs
            while stream.active and not self.interruption_event.is_set():
                time.sleep(0.1)

            if self.interruption_event.is_set():
                logging.warning("Playback interrupted by user.")
                stream.stop()
                stream.close()
            else:
                # If not interrupted, wait for the stream to naturally finish
                stream.stop()
                stream.close()

        except Exception as e:
            logging.error(f"Error during audio playback: {e}", exc_info=True)
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception as close_e:
                    logging.error(f"Error closing stream: {close_e}")
        finally:
            is_speaking = False
            logging.info("Playback thread finished.")

# --- Audio Input Callback ---
def audio_callback(indata, frames, time, status):
    """Called by sounddevice for each audio block."""
    global vad_buffer
    if status:
        logging.warning(f"Audio callback status: {status}")

    # We will now queue audio even if the agent is speaking to detect interruptions.
    if not agent_busy:
        audio_bytes = indata.tobytes()
        audio_queue.put(audio_bytes)

        # Process audio for VAD-based interruption detection
        if is_speaking:
            vad_buffer.extend(audio_bytes)
            while len(vad_buffer) >= VAD_BYTES_PER_FRAME:
                frame = vad_buffer[:VAD_BYTES_PER_FRAME]
                vad_buffer = vad_buffer[VAD_BYTES_PER_FRAME:]
                try:
                    if vad.is_speech(frame, SAMPLE_RATE):
                        logging.info("Interruption detected by VAD!")
                        if not interruption_event.is_set():
                            interruption_event.set()
                        # Once detected, we can stop checking and clear the buffer
                        vad_buffer.clear()
                        break 
                except Exception as e:
                    logging.error(f"VAD processing error: {e}")


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

def _clear_audio_queue():
    """Remove any pending audio frames."""
    with audio_queue.mutex:
        audio_queue.queue.clear()
    global vad_buffer
    vad_buffer.clear()

async def handle_agent_response(transcript, stt_end_time):
    """Gets response from LLM, synthesizes audio, and plays it."""
    global agent_busy, current_playback_thread
    
    if agent_busy:
        logging.warning("Agent is already busy, skipping new request.")
        return

    agent_busy = True
    interruption_event.clear() # Clear any previous interruption flags
    _clear_audio_queue()
    print("\n--- AGENT'S TURN (Please wait) ---")
    logging.info("User finished speaking. Agent is processing...")

    try:
        # --- 1. LLM Query ---
        logging.info(f"Querying agent with: '{transcript}'")
        llm_start_time = time.time()
        agent_response = await asyncio.to_thread(agent.query, input=transcript)
        llm_end_time = time.time()
        tts_text = _to_text(agent_response)
        logging.info(f"LATENCY: LLM query took {llm_end_time - llm_start_time:.2f} seconds.")
        
        logging.info(f"Agent says: {tts_text}")
        chat_history.append(HumanMessage(content=transcript))
        chat_history.append(AIMessage(content=tts_text))
        
        if not tts_text or not tts_text.strip():
            logging.warning("TTS text is empty; skipping audio generation.")
            agent_busy = False
            return

        # --- 2. TTS Synthesis ---
        _clear_audio_queue() # Clear any audio captured during LLM thinking time
        logging.info("Generating audio response from Google TTS...")
        tts_start_time = time.time()
        synthesis_input = texttospeech.SynthesisInput(text=tts_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Standard-C"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
        )

        response = await asyncio.to_thread(
            tts_client.synthesize_speech,
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        tts_end_time = time.time()
        logging.info(f"LATENCY: TTS synthesis took {tts_end_time - tts_start_time:.2f} seconds.")
        
        # --- 3. Audio Playback in a separate thread ---
        samples_i16 = np.frombuffer(response.audio_content, dtype=np.int16)
        current_playback_thread = PlaybackThread(samples_i16, SAMPLE_RATE, interruption_event)
        current_playback_thread.start()
        
    except Exception as e:
        logging.error(f"Error in handle_agent_response: {e}", exc_info=True)
    finally:
        agent_busy = False
        # The main loop will now handle the "YOUR TURN TO SPEAK" logic
        # based on the playback thread's status.

async def google_stt_recognizer():
    """Manages the streaming recognition with Google Cloud STT."""
    while True:
        try:
            client, streaming_config = get_stt_config()
            
            loop = asyncio.get_running_loop()

            def audio_generator():
                """Yields audio chunks from the queue."""
                while True:
                    chunk = audio_queue.get()
                    if chunk is None:
                        break
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            async def process_responses_async():
                """Async wrapper to process blocking STT responses."""
                stt_started = False
                stt_start_time = None
                
                try:
                    responses = await asyncio.to_thread(
                        client.streaming_recognize,
                        config=streaming_config,
                        requests=audio_generator(),
                        timeout=240  # Add a timeout to prevent indefinite blocking
                    )

                    for response in responses:
                        if not response.results:
                            continue

                        result = response.results[0]
                        if not result.alternatives:
                            continue

                        transcript = result.alternatives[0].transcript

                        if not stt_started and transcript:
                            stt_start_time = time.time()
                            stt_started = True

                        if result.is_final:
                            stt_end_time = time.time()
                            logging.info(f"\n>>> You said (final): {transcript}")
                            if stt_start_time:
                                logging.info(f"LATENCY: STT processing took {stt_end_time - stt_start_time:.2f}s from start of utterance.")
                            if transcript.strip():
                                # If an interruption happened, stop the current playback.
                                if current_playback_thread and current_playback_thread.is_alive():
                                    logging.info("Processing interrupted speech, stopping current playback.")
                                    interruption_event.set()
                                    # Give the thread a moment to stop
                                    await asyncio.sleep(0.1)

                                logging.info("Final transcript received, creating agent task.")
                                asyncio.create_task(handle_agent_response(transcript, stt_end_time))
                            
                            # The stream is now finished because of single_utterance=True
                            return # Exit this inner function to start a new recognition stream

                except Exception as e:
                    if "Deadline Exceeded" in str(e) or "Audio Timeout" in str(e):
                        logging.warning(f"STT stream timed out: {e}. Reconnecting...")
                    else:
                        logging.error(f"STT worker error: {e}", exc_info=True)
                finally:
                    # This will be hit when the stream ends (due to single_utterance or timeout)
                    logging.info("STT stream ended. Will restart for next utterance.")


            # This will block until the first utterance is finalized
            await process_responses_async()
            
            # After an utterance, clear any lingering audio and wait a moment
            _clear_audio_queue()
            await asyncio.sleep(0.1)

        except Exception as e:
            logging.error(f"STT recognizer loop error: {e}", exc_info=True)
            await asyncio.sleep(1) # Avoid rapid-fire reconnection on persistent errors

def get_stt_config():
    """Returns the Google STT client and streaming configuration."""
    # Setting single_utterance=True means the stream will automatically close
    # after the first utterance is detected and finalized.
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        model="telephony",
        use_enhanced=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True, single_utterance=True
    )

    logging.info("Connecting to Google STT stream...")
    client = speech.SpeechClient()
    return client, streaming_config

async def main():
    """Main function to run the local voice agent test."""
    logging.info("Starting local voice agent test...")
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logging.error("The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        logging.error("Please follow the setup instructions in GOOGLE_API_SETUP.md.")
        return

    try:
        # Start the STT recognizer task
        stt_task = asyncio.create_task(google_stt_recognizer())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):
            print("\n--- YOUR TURN TO SPEAK ---")
            logging.info("Speak into your microphone now. The agent will respond when you pause.")
            
            # Main loop to manage state and print prompts
            while True:
                if current_playback_thread and current_playback_thread.is_alive():
                    # Agent is speaking or has just finished
                    await asyncio.sleep(0.1)
                    continue
                
                if not agent_busy and (not current_playback_thread or not current_playback_thread.is_alive()):
                    if interruption_event.is_set():
                        # Don't print "Your turn" if an interruption just happened,
                        # as a new agent response is likely imminent.
                        pass
                    else:
                        # This state is after playback is done and no new task is running
                        # print("\n--- YOUR TURN TO SPEAK ---")
                        # logging.info("Agent finished speaking. Resuming listening.")
                        pass # The prompt is now handled by the STT loop logic
                
                await asyncio.sleep(0.1)


    except Exception as e:
        logging.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        if current_playback_thread and current_playback_thread.is_alive():
            interruption_event.set()
            current_playback_thread.join()
        audio_queue.put(None)
        logging.info("Test finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping test...")
    except Exception as e:
        logging.error(f"Unhandled exception in top-level: {e}", exc_info=True)
