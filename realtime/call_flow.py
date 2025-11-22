from __future__ import annotations

import asyncio
import audioop
import base64
import logging
from typing import Any, Callable, Coroutine, Dict, Final, Optional

from google.genai import types as genai_types

import config
from google_calendar import book_appointment, find_available_slots
from realtime.gemini_live_client import GeminiLiveClient

logger = logging.getLogger(__name__)

# --- Tool Registry ---
# Maps tool names to their implementation and required arguments.
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "find_available_slots": {
        "function": find_available_slots,
        "required_args": ["date_str"],
        "arg_map": {"date": "date_str"},  # Maps LLM argument 'date' to function's 'date_str'
    },
    "book_appointment": {
        "function": book_appointment,
        "required_args": ["datetime_str", "full_name", "email", "property_id"],
        "arg_map": {},
    },
}


class CallFlowManager:
    """
    Manages the real-time, bidirectional conversation between a user (on a Twilio call)
    and the Gemini Live API. It handles:
    - Streaming audio between Twilio and Gemini.
    - Processing text responses from Gemini (both interim and final).
    - Executing tool calls requested by Gemini (e.g., checking a calendar).
    - Sending tool responses back to Gemini.
    - Gracefully handling connection errors from the Gemini client.
    """

    def __init__(
        self,
        call_sid: str,
        stream_sid: str,
        twilio_ws_send_callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.twilio_ws_send: Final = twilio_ws_send_callback
        self.gemini_client: Optional[GeminiLiveClient] = None
        self.is_active = False

    async def start(self) -> None:
        """Initializes the Gemini client and starts the call flow."""
        if self.is_active:
            logger.warning("Attempted to start an already active call flow for call=%s", self.call_sid)
            return
            
        logger.info("Starting Gemini Live call flow for call=%s", self.call_sid)
        self.is_active = True
        
        self.gemini_client = GeminiLiveClient(
            on_audio_callback=self._handle_gemini_audio,
            on_text_callback=self._handle_gemini_text,
            on_tool_call_callback=self._handle_tool_call,
            on_error_callback=self._handle_gemini_error,
            on_user_text_callback=self._handle_caller_text,
        )
        
        try:
            await self.gemini_client.connect()
            logger.info("Call flow started and Gemini client connected for call=%s", self.call_sid)
        except Exception as e:
            logger.critical(
                "Failed to connect to Gemini during call start for call=%s. Error: %s",
                self.call_sid, e, exc_info=True
            )
            # If connection fails on startup, immediately stop the flow.
            await self.stop()

    async def stop(self) -> None:
        """Stops the call flow and disconnects the Gemini client."""
        if not self.is_active:
            # This can happen if stop() is called multiple times, which is fine.
            return
            
        logger.info("Stopping Gemini Live call flow for call=%s", self.call_sid)
        self.is_active = False
        
        if self.gemini_client:
            await self.gemini_client.disconnect()
            
        logger.info("Call flow stopped for call=%s", self.call_sid)

    async def handle_audio_from_twilio(self, audio_data: bytes) -> None:
        """Receives audio from Twilio and forwards it to Gemini."""
        if not self.is_active or not self.gemini_client:
            return
        await self.gemini_client.send_audio(audio_data)

    async def _handle_gemini_error(self, error: Exception) -> None:
        """Handles a critical error from the Gemini client by stopping the call."""
        logger.error(
            "Critical error received from Gemini client for call=%s. Stopping call flow. Error: %s",
            self.call_sid, error, exc_info=True
        )
        # Trigger a graceful shutdown of the entire call flow.
        await self.stop()

    def _resample_audio(self, pcm_audio: bytes, source_rate: int, target_rate: int) -> bytes:
        """Resamples PCM audio from a source rate to a target rate."""
        if source_rate == target_rate:
            return pcm_audio
        
        resampled_audio, _ = audioop.ratecv(
            pcm_audio,
            2,  # 2 bytes per sample (16-bit)
            1,  # Mono channel
            source_rate,
            target_rate,
            None,
        )
        return resampled_audio

    async def _handle_gemini_audio(self, audio_data: bytes) -> None:
        """
        Receives audio from Gemini, resamples it for Twilio,
        converts it to mu-law, and sends it to Twilio.
        """
        # Resample from Gemini's sample rate to Twilio's required 8000 Hz
        resampled_pcm = self._resample_audio(
            pcm_audio=audio_data,
            source_rate=config.GEMINI_AUDIO_SAMPLE_RATE,
            target_rate=8000,
        )
        
        # Convert 16-bit linear PCM to 8-bit mu-law
        mulaw_audio = audioop.lin2ulaw(resampled_pcm, 2)
        
        # Base64 encode the payload for the Twilio Media Stream message
        payload = base64.b64encode(mulaw_audio).decode("utf-8")
        
        await self.twilio_ws_send(
            {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            }
        )

    async def _handle_gemini_text(self, text: str) -> None:
        """Logs the agent's transcribed speech."""
        text = text.strip()
        if text:
            logger.info("Agent: %s", text)

    async def _handle_caller_text(self, text: str) -> None:
        """Logs the caller's transcribed speech."""
        text = text.strip()
        if text:
            logger.info("Caller: %s", text)

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """
        Handles a tool call request from Gemini by dispatching it to the
        appropriate function and sending the result back.
        """
        # Extract function calls, supporting different potential structures
        if isinstance(tool_call, genai_types.LiveServerToolCall):
            function_calls = tool_call.function_calls or []
        elif isinstance(tool_call, Dict):
            function_calls = tool_call.get("functionCalls") or tool_call.get("function_calls") or []
        else:
            function_calls = []

        # It's possible to receive multiple tool calls in one message
        for call in function_calls:
            call_id = call.get("id")
            function_name = call.get("name")
            args = call.get("args", {})

            logger.info(
                "Gemini requested tool '%s' for call %s with args=%s",
                function_name, self.call_sid, args
            )

            # Execute the tool and get the response payload
            response_payload = await self._execute_tool(function_name, args)

            # Send the result back to Gemini
            if call_id and self.gemini_client and self.is_active:
                logger.info("Sending tool response for '%s' to Gemini.", function_name)
                await self.gemini_client.send_tool_response(
                    tool_call_id=call_id,
                    result={
                        "name": function_name,
                        "response": response_payload,
                    },
                )

    async def _execute_tool(self, name: Optional[str], args: Dict[str, Any]) -> str:
        """
        Executes a tool from the TOOL_REGISTRY.
        - Validates required arguments.
        - Runs the synchronous tool function in a separate thread to avoid blocking.
        - Catches exceptions and returns a user-friendly error message.
        """
        if not name or name not in TOOL_REGISTRY:
            return f"Error: Unknown tool '{name}'."

        tool_config = TOOL_REGISTRY[name]
        tool_function = tool_config["function"]
        required_args = tool_config["required_args"]
        arg_map = tool_config["arg_map"]

        # Map LLM args to function args and check for missing ones
        final_args = {}
        missing_args = []
        for req_arg in required_args:
            # Find the argument name the LLM might have used
            llm_arg_name = next((k for k, v in arg_map.items() if v == req_arg), req_arg)
            
            if llm_arg_name in args:
                final_args[req_arg] = args[llm_arg_name]
            else:
                missing_args.append(req_arg)

        if missing_args:
            error_msg = f"Error: Missing required arguments for '{name}': {', '.join(missing_args)}."
            logger.error(error_msg)
            return error_msg

        try:
            # Run the synchronous tool function in a non-blocking way
            logger.info("Executing tool '%s' with args: %s", name, final_args)
            result = await asyncio.to_thread(tool_function, **final_args)
            logger.info("Tool '%s' executed successfully. Result: %s", name, result)
            return str(result)
        except Exception as e:
            logger.error(
                "An error occurred while executing tool '%s': %s", name, e, exc_info=True
            )
            return f"Error: An unexpected error occurred while trying to execute {name}."
