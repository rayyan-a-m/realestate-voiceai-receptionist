from __future__ import annotations

import asyncio
import audioop
import logging
import re
from contextlib import suppress
from typing import Any, AsyncContextManager, Awaitable, Callable, Dict, Optional

from google import genai
from google.api_core import exceptions
from google.genai import types as genai_types
from google.genai.live import AsyncSession

import config
from realtime.gemini_config import build_live_connect_config

logger = logging.getLogger(__name__)

# --- Type Definitions for Callbacks ---
AudioCallback = Callable[[bytes], Awaitable[None]]
TextCallback = Callable[[str], Awaitable[None]]
ToolCallback = Callable[[genai_types.LiveServerToolCall], Awaitable[None]]
ErrorCallback = Callable[[Exception], Awaitable[None]]


class GeminiLiveClient:
    """
    A robust wrapper for the Google GenAI SDK's Gemini Live (real-time) conversation API.

    This class manages the WebSocket connection, handles automatic reconnections,
    processes incoming/outgoing audio and text, and manages the session lifecycle.
    """

    def __init__(
        self,
        on_audio_callback: AudioCallback,
        on_text_callback: TextCallback,
        on_tool_call_callback: ToolCallback,
        on_error_callback: ErrorCallback,
        on_user_text_callback: Optional[TextCallback] = None,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: int = 2,
    ) -> None:
        # Callbacks
        self.on_audio_callback = on_audio_callback
        self.on_text_callback = on_text_callback
        self.on_tool_call_callback = on_tool_call_callback
        self.on_error_callback = on_error_callback
        self.on_user_text_callback = on_user_text_callback

        # State
        self.client: Optional[genai.Client] = None
        self.session_cm: Optional[AsyncContextManager[AsyncSession]] = None
        self.session: Optional[AsyncSession] = None
        self.receive_task: Optional[asyncio.Task[None]] = None
        self.is_connected = False
        self._is_disconnecting = False

        # Reconnection logic
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

    async def connect(self) -> None:
        """
        Establishes a connection to the Gemini Live service with a retry mechanism.
        """
        if self.is_connected:
            logger.warning("Connect called on an already connected client.")
            return

        self._is_disconnecting = False
        
        for attempt in range(self.reconnect_attempts + 1):
            try:
                client_kwargs = self._client_kwargs()
                logger.info(f"Connecting to Gemini Live (Attempt {attempt + 1}/{self.reconnect_attempts + 1})...")

                self.client = genai.Client(**client_kwargs)
                live_config = build_live_connect_config()

                self.session_cm = self.client.aio.live.connect(
                    model=config.GEMINI_LIVE_MODEL,
                    config=live_config,
                )
                
                self.session = await self.session_cm.__aenter__()
                self.is_connected = True
                self.receive_task = asyncio.create_task(self._receive_messages())
                
                logger.info("Gemini Live session established successfully.")
                return  # Exit on successful connection

            except Exception as e:
                logger.error(f"Failed to connect to Gemini Live on attempt {attempt + 1}: {e}", exc_info=True)
                await self._close_session() # Ensure cleanup before retry
                if attempt < self.reconnect_attempts:
                    logger.info(f"Retrying in {self.reconnect_delay_seconds} seconds...")
                    await asyncio.sleep(self.reconnect_delay_seconds)
                else:
                    logger.critical("All reconnection attempts failed. Raising final exception.")
                    await self.on_error_callback(e)
                    raise

    async def disconnect(self) -> None:
        """Gracefully disconnects the Gemini Live session."""
        if not self.is_connected or self._is_disconnecting:
            return
        
        logger.info("Disconnecting from Gemini Live session...")
        self._is_disconnecting = True
        self.is_connected = False

        if self.session:
            with suppress(Exception):
                # Signal the end of the audio stream to the server
                await self.session.send_realtime_input(audio_stream_end=True)

        if self.receive_task:
            self.receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.receive_task

        await self._close_session()
        logger.info("Gemini Live session closed.")

    async def send_audio(self, audio_data: bytes) -> None:
        """Processes and sends mulaw audio from Twilio to Gemini Live."""
        if not self.is_connected:
            logger.warning("Not connected; dropping audio chunk.")
            return

        try:
            # Convert 8-bit mu-law to 16-bit linear PCM
            pcm_audio = audioop.ulaw2lin(audio_data, 2)
            # Resample from Twilio's 8kHz to Gemini's required sample rate
            resampled_audio = self._resample_pcm(pcm_audio, 8000, config.GEMINI_AUDIO_SAMPLE_RATE)
            await self._send_pcm_frame(resampled_audio, config.GEMINI_AUDIO_SAMPLE_RATE)
        except Exception as e:
            logger.error(f"Failed to send audio to Gemini Live: {e}", exc_info=True)
            await self.on_error_callback(e)

    async def send_tool_response(self, tool_call_id: str, result: Dict[str, Any]) -> None:
        """Sends the result of a tool execution back to Gemini."""
        if not self.is_connected:
            logger.warning("Not connected; cannot send tool response.")
            return

        function_response = genai_types.FunctionResponse(
            id=tool_call_id,
            name=result.get("name"),
            response={"content": result.get("response")}, # Ensure response is nested under 'content'
        )

        try:
            await self.session.send_tool_response(function_responses=[function_response])
            logger.info(f"Successfully sent tool response for call ID: {tool_call_id}")
        except Exception as e:
            logger.error(f"Failed to send tool response to Gemini Live: {e}", exc_info=True)
            await self.on_error_callback(e)

    async def _send_pcm_frame(self, pcm_audio: bytes, sample_rate: int) -> None:
        """Sends a raw PCM audio frame to the Gemini session."""
        if not self.session: return
        
        blob = genai_types.Blob(
            data=pcm_audio,
            mime_type=f"audio/pcm;rate={sample_rate}",
        )
        await self.session.send_realtime_input(media=blob)

    async def _receive_messages(self) -> None:
        """The main loop for receiving and handling messages from the Gemini server."""
        assert self.session is not None
        try:
            async for message in self.session.receive():
                if self._is_disconnecting: break
                await self._handle_server_message(message)
        except (asyncio.CancelledError, exceptions.Cancelled):
            logger.info("Receive loop cancelled as part of disconnection.")
        except exceptions.DeadlineExceeded:
            logger.error("Gemini Live connection timed out (DeadlineExceeded). Attempting to reconnect...")
            await self.on_error_callback(exceptions.DeadlineExceeded("Connection timed out."))
        except Exception as e:
            logger.error(f"Gemini Live receive loop exited unexpectedly: {e}", exc_info=True)
            if not self._is_disconnecting:
                await self.on_error_callback(e)
        finally:
            if not self._is_disconnecting:
                logger.warning("Receive loop ended prematurely. Cleaning up connection.")
                await self.disconnect()

    async def _handle_server_message(self, message: genai_types.LiveServerMessage) -> None:
        """Routes incoming server messages to the appropriate handlers."""
        if message.server_content:
            await self._handle_server_content(message.server_content)
        if message.tool_call:
            await self.on_tool_call_callback(message.tool_call)

    async def _handle_server_content(self, content: genai_types.LiveServerContent) -> None:
        """Handles content parts from the model, like audio and text."""
        if model_turn := content.model_turn:
            for part in model_turn.parts or []:
                if part.inline_data and part.inline_data.data:
                    await self._handle_audio_part(part.inline_data)
                if part.text:
                    await self.on_text_callback(part.text)

        if (transcription := content.output_transcription) and transcription.text:
            await self.on_text_callback(transcription.text)

        if self.on_user_text_callback and (transcription := content.input_transcription) and transcription.text:
            await self.on_user_text_callback(transcription.text)

    async def _handle_audio_part(self, blob: genai_types.Blob) -> None:
        """Handles an audio part, resampling if necessary, and passes it to the callback."""
        pcm_audio = blob.data or b""
        source_rate = self._extract_sample_rate(blob.mime_type)
        
        resampled_audio = self._resample_pcm(pcm_audio, source_rate, config.GEMINI_AUDIO_SAMPLE_RATE)
        await self.on_audio_callback(resampled_audio)

    def _client_kwargs(self) -> Dict[str, Any]:
        """Constructs the keyword arguments for the GenAI client."""
        if config.GCP_PROJECT_ID:
            if not config.GCP_LOCATION:
                raise ValueError("GCP_LOCATION is required for Vertex Gemini Live.")
            return {"vertexai": True, "project": config.GCP_PROJECT_ID, "location": config.GCP_LOCATION}
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when no GCP project is set.")
        return {"api_key": config.GOOGLE_API_KEY}

    async def _close_session(self) -> None:
        """Safely closes the session context manager."""
        if self.session_cm:
            with suppress(Exception):
                await self.session_cm.__aexit__(None, None, None)
        self.session_cm = None
        self.session = None

    @staticmethod
    def _resample_pcm(pcm_audio: bytes, source_rate: int, target_rate: int) -> bytes:
        """Resamples PCM audio data if the source and target rates differ."""
        if source_rate == target_rate:
            return pcm_audio
        
        resampled_audio, _ = audioop.ratecv(pcm_audio, 2, 1, source_rate, target_rate, None)
        return resampled_audio

    @staticmethod
    def _extract_sample_rate(mime_type: Optional[str]) -> int:
        """Extracts the sample rate from an audio MIME type string."""
        if not mime_type:
            return config.GEMINI_AUDIO_SAMPLE_RATE
        if match := re.search(r"rate=(\d+)", mime_type):
            return int(match.group(1))
        return config.GEMINI_AUDIO_SAMPLE_RATE

