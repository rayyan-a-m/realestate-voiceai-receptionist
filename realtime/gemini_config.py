from __future__ import annotations

from typing import Any, Dict, List

import config
from prompts import SYSTEM_PROMPT

# Tool schemas exposed to Gemini Live so it knows how to interact with
# our calendar helpers. The names MUST exactly match the callable names
# we handle inside CallFlowManager.
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "find_available_slots",
        "description": (
            "Look up available 30-minute appointment slots on a specific "
            "date. Always ask the caller for the exact date before invoking "
            "this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Desired appointment date in YYYY-MM-DD format."
                }
            },
            "required": ["date"],
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book the confirmed property tour. Only call after the caller has "
            "picked a slot returned by find_available_slots and provided their "
            "full details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "datetime_str": {
                    "type": "string",
                    "description": "Confirmed appointment time in 'YYYY-MM-DD HH:MM' format."
                },
                "full_name": {
                    "type": "string",
                    "description": "Caller full name."
                },
                "email": {
                    "type": "string",
                    "description": "Caller email."
                },
                "property_id": {
                    "type": "string",
                    "description": "Internal property identifier (e.g., prop123)."
                }
            },
            "required": ["datetime_str", "full_name", "email", "property_id"],
        },
    },
]


def build_gemini_setup() -> Dict[str, Any]:
    """Return the setup payload for the Gemini Live WebSocket."""

    model_name = config.GEMINI_LIVE_MODEL
    sample_rate = config.GEMINI_AUDIO_SAMPLE_RATE

    return {
        "model": model_name,
        "response": {"modalities": ["audio"]},
        "resource": {
            "audio": {
                "sample_rate_hz": sample_rate,
                "encoding": "pcm16",
            }
        },
        "instructions": [
            {
                "role": "system",
                "parts": [
                    {
                        "text": SYSTEM_PROMPT,
                    }
                ],
            }
        ],
        "tools": [
            {
                "function_declarations": TOOL_DEFINITIONS,
            }
        ],
    }


def build_live_connect_config() -> Dict[str, Any]:
    """Construct the LiveConnectConfig as a plain dict for the google-genai SDK."""

    config_payload: Dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "system_instruction": {
            "role": "system",
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "tools": [
            {
                "function_declarations": TOOL_DEFINITIONS,
            }
        ],
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {
                "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
                "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
            }
        },
    }

    if config.GEMINI_VOICE_NAME and "native-audio" in config.GEMINI_LIVE_MODEL:
        config_payload["speech_config"] = {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": config.GEMINI_VOICE_NAME,
                }
            }
        }

    return config_payload
