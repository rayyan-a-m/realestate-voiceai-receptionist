"""
Harness for running the AI Voice Agent locally using your microphone and speakers.
This script simulates the full end-to-end flow without needing a live phone call.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import logging
from typing import Any, Dict

import sounddevice as sd

import config
from realtime.call_flow import CallFlowManager

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Audio settings - Must match your microphone's capabilities
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_DTYPE = "int16"
# How many milliseconds of audio to capture in each chunk
CHUNK_MS = 100
FRAMES_PER_CHUNK = int(MIC_SAMPLE_RATE * (CHUNK_MS / 1000.0))


class LocalAudioDevice:
    """
    A helper class to manage local microphone input and speaker output
    using the sounddevice library.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        
        self._input_stream = sd.RawInputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype=MIC_DTYPE,
            blocksize=FRAMES_PER_CHUNK,
            callback=self._on_mic_audio,
        )
        self._output_stream = sd.RawOutputStream(
            samplerate=config.GEMINI_AUDIO_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype=MIC_DTYPE,
        )

    def start(self) -> None:
        """Starts the audio input and output streams."""
        logger.info("Starting local audio devices...")
        self._output_stream.start()
        self._input_stream.start()
        logger.info("Microphone is now live.")

    def stop(self) -> None:
        """Stops and closes the audio streams."""
        logger.info("Stopping local audio devices...")
        with contextlib.suppress(Exception):
            self._input_stream.stop()
            self._input_stream.close()
        with contextlib.suppress(Exception):
            self._output_stream.stop()
            self._output_stream.close()

    def _on_mic_audio(self, indata: bytes, frames: int, time_info: Any, status: Any) -> None:
        """Callback executed by sounddevice for each chunk of microphone audio."""
        if status:
            logger.warning("Microphone stream status: %s", status)
        # This callback is run in a separate thread, so we use thread-safe methods.
        self._loop.call_soon_threadsafe(self._input_queue.put_nowait, indata)

    async def get_audio_chunk(self) -> bytes:
        """Retrieves the next chunk of audio from the microphone."""
        return await self._input_queue.get()

    def play_audio_chunk(self, audio_bytes: bytes) -> None:
        """Plays a chunk of audio on the local speakers."""
        self._output_stream.write(audio_bytes)


async def main() -> None:
    """
    Main function to set up and run the local test environment.
    """
    logger.info("Starting Real Estate AI Voice Agent - Local Test Harness")
    
    # Validate configuration before starting
    try:
        config.validate_config()
    except ValueError as e:
        logger.critical(f"Configuration error: {e}")
        return

    loop = asyncio.get_running_loop()
    audio_device = LocalAudioDevice(loop)

    try:
        audio_device.start()
    except sd.PortAudioError as e:
        logger.critical(f"Failed to start audio devices: {e}. Please check your microphone/speaker configuration.")
        return

    # This callback simulates the behavior of the FastAPI WebSocket, which sends
    # audio from our system back to Twilio. Here, we just play it on the speakers.
    async def mock_twilio_ws_send(message: Dict[str, Any]) -> None:
        if message["event"] == "media":
            # The payload is base64 encoded mu-law audio. We need to decode it.
            payload_b64 = message["media"]["payload"]
            # Decode from base64
            mulaw_audio = base64.b64decode(payload_b64)
            # Convert from 8-bit mu-law back to 16-bit linear PCM for playback
            pcm_audio = audioop.ulaw2lin(mulaw_audio, 2)
            # Play the audio on the local speakers
            audio_device.play_audio_chunk(pcm_audio)

    # Instantiate the exact same CallFlowManager used in the main application.
    # This is key to ensuring our test is accurate.
    call_flow = CallFlowManager(
        call_sid="local_test_call",
        stream_sid="local_test_stream",
        twilio_ws_send_callback=mock_twilio_ws_send,
    )

    try:
        # Start the call flow, which connects to Gemini.
        await call_flow.start()
        logger.info("Agent is ready. Speak into your microphone to begin the conversation.")

        # Main loop: continuously read from the microphone and send to the call flow.
        while call_flow.is_active:
            audio_chunk = await audio_device.get_audio_chunk()
            
            # In the main app, this audio comes from Twilio. Here, it comes from the mic.
            # We need to convert it to mu-law format, just like Twilio does.
            mulaw_chunk = audioop.lin2ulaw(audio_chunk, 2)
            
            # The CallFlowManager expects base64 encoded audio bytes, just like it
            # would receive from the FastAPI/Twilio WebSocket.
            b64_encoded_chunk = base64.b64encode(mulaw_chunk)
            
            # This simulates the `handle_audio_from_twilio` call in the real app.
            await call_flow.handle_audio_from_twilio(b64_encoded_chunk)

    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        audio_device.stop()
        if call_flow.is_active:
            await call_flow.stop()
        logger.info("Local test harness stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)
