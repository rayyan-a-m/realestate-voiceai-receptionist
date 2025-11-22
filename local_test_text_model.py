"""Text-input harness for exercising the Gemini Live agent and its tools."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Optional

import config
from realtime.call_flow import CallFlowManager
from realtime.gemini_live_client import GeminiLiveClient

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency safeguard
    sd = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEXT_FLUSH_DELAY_SECONDS = 0.35


def _merge_text_chunks(chunks: list[str]) -> str:
    """Merges a list of text chunks into a single, cleaned-up string."""
    raw = " ".join(part.strip() for part in chunks if part.strip())
    if not raw:
        return ""
    # Fix spacing around punctuation
    raw = re.sub(r"\s+([,.;!?])", r"\1", raw)
    # Consolidate multiple spaces
    raw = re.sub(r"\s{2,}", " ", raw)
    return raw.strip()


async def handle_tool_call(
    tool_call: Dict[str, Any],
    client: GeminiLiveClient,
    call_flow: CallFlowManager,
) -> None:
    """
    Handles a tool call request from the Gemini model by delegating
    to the CallFlowManager's tool execution logic.
    """
    function_calls = tool_call.get("functionCalls") or tool_call.get("function_calls") or []

    for call in function_calls:
        call_id = call.get("id")
        function_name = call.get("name")
        args = call.get("args", {})

        if not function_name:
            logger.warning("Received tool call with no function name: %s", call)
            continue

        logger.info("Agent requested tool '%s' with args=%s", function_name, args)
        
        # Delegate the actual execution to the CallFlowManager instance
        response_payload = await call_flow._execute_tool(function_name, args)

        if call_id:
            logger.info("Sending tool response for call_id '%s'", call_id)
            await client.send_tool_response(
                tool_call_id=call_id,
                result={"name": function_name, "response": response_payload},
            )


async def main() -> None:
    """Main function to set up and run the text-based test environment."""
    logger.info("Starting Real Estate AI Voice Agent - Text-Based Test Harness")

    # Validate configuration before starting
    try:
        config.validate_config()
    except ValueError as e:
        logger.critical(f"Configuration error: {e}")
        return

    loop = asyncio.get_running_loop()
    client: Optional[GeminiLiveClient] = None
    text_chunks: list[str] = []
    flush_handle: Optional[asyncio.Handle] = None
    audio_stream: Optional[Any] = None

    # --- Audio Output Setup (Optional) ---
    def start_audio_output() -> None:
        nonlocal audio_stream
        if sd is None:
            logger.warning("sounddevice is not installed; audio playback is disabled.")
            return
        if audio_stream is not None:
            return
        try:
            audio_stream = sd.RawOutputStream(
                samplerate=config.GEMINI_AUDIO_SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            audio_stream.start()
            logger.info("Audio output stream started.")
        except Exception as exc:
            audio_stream = None
            logger.warning(f"Failed to start audio playback: {exc}")

    def stop_audio_output() -> None:
        nonlocal audio_stream
        if not audio_stream:
            return
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            logger.exception("Error stopping audio playback")
        finally:
            audio_stream = None
            logger.info("Audio output stream stopped.")

    # --- Text Output Handling ---
    async def flush_text_buffer() -> None:
        nonlocal text_chunks, flush_handle
        if not text_chunks:
            return
        message = _merge_text_chunks(text_chunks)
        text_chunks = []
        flush_handle = None
        if message:
            print(f"Agent > {message}")

    def schedule_text_flush() -> None:
        nonlocal flush_handle
        if flush_handle:
            flush_handle.cancel()
        flush_handle = loop.call_later(
            TEXT_FLUSH_DELAY_SECONDS,
            lambda: asyncio.create_task(flush_text_buffer()),
        )

    # --- Gemini Client Callbacks ---
    async def on_audio_chunk(audio_bytes: bytes) -> None:
        if not audio_bytes or audio_stream is None:
            return
        await loop.run_in_executor(None, audio_stream.write, audio_bytes)

    async def on_text_output(text: str) -> None:
        if text_stripped := text.strip():
            text_chunks.append(text_stripped)
            schedule_text_flush()

    async def on_user_transcript(text: str) -> None:
        if text_stripped := text.strip():
            # This is the transcription of the text we sent, so we just log it.
            logger.info("Transcription of sent text: '%s'", text_stripped)

    # Instantiate CallFlowManager solely for its tool execution logic.
    # We don't need its audio handling capabilities for this text-based test.
    call_flow = CallFlowManager(
        call_sid="local_text_test",
        stream_sid="local_text_test",
        twilio_ws_send_callback=None,  # No Twilio connection
    )

    async def on_tool_request(tool_call: Dict[str, Any]) -> None:
        if client is None:
            logger.warning("Received tool request before client initialization")
            return
        # Delegate to our new handler which uses the CallFlowManager
        await handle_tool_call(tool_call, client, call_flow)

    # --- Main Application Logic ---
    client = GeminiLiveClient(
        on_audio_callback=on_audio_chunk,
        on_text_callback=on_text_output,
        on_tool_call_callback=on_tool_request,
        on_user_text_callback=on_user_transcript,
    )

    try:
        start_audio_output()
        await client.connect()
    except Exception as exc:
        logger.error(f"Failed to connect to Gemini Live: {exc}")
        stop_audio_output()
        return

    print("\nAgent is ready. Type your message and press Enter.")
    print("Type 'quit' or 'exit' to end the session.\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You   > ")
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input.strip():
                continue
            
            await client.send_text(user_input)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("User interrupted. Shutting down.")
    except Exception as exc:
        logger.error(f"An error occurred in the main loop: {exc}", exc_info=True)
    finally:
        if flush_handle:
            flush_handle.cancel()
        await flush_text_buffer()
        
        logger.info("Cleaning up resources...")
        stop_audio_output()
        if client and client.is_connected:
            await client.disconnect()
        logger.info("Text-based test harness stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This is handled gracefully in the main loop now
        pass
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)
