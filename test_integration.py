import asyncio
import argparse
import logging
from deepgram import DeepgramClient, PrerecordedOptions
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialization ---
try:
    deepgram_client = DeepgramClient(config.DEEPGRAM_API_KEY)
    elevenlabs_client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize clients: {e}")
    exit(1)

async def main(input_file_path: str):
    """
    Tests the Deepgram and ElevenLabs integration.
    1. Transcribes the given audio file using Deepgram.
    2. Converts the transcript back to audio using ElevenLabs.
    3. Saves the generated audio to 'output.mp3'.
    """
    try:
        # --- 1. Transcribe with Deepgram ---
        logging.info(f"Opening audio file: {input_file_path}")
        with open(input_file_path, 'rb') as audio_file:
            buffer_data = audio_file.read()

        payload = {'buffer': buffer_data}
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        logging.info("Sending audio to Deepgram for transcription...")
        response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
        
        transcript = response.results.channels[0].alternatives[0].transcript
        if not transcript:
            logging.warning("Transcription is empty. Exiting.")
            return
            
        logging.info(f"Successfully transcribed text: '{transcript}'")

        # --- 2. Synthesize with ElevenLabs ---
        logging.info("Sending transcribed text to ElevenLabs for speech synthesis...")
        
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=transcript,
            voice_id=config.ELEVENLABS_VOICE_ID,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
        )

        # --- 3. Save the output audio ---
        output_file_path = "output.mp3"
        logging.info(f"Streaming generated audio to {output_file_path}...")
        with open(output_file_path, 'wb') as output_file:
            async for chunk in audio_stream:
                if chunk:
                    output_file.write(chunk)
        
        logging.info(f"Audio successfully saved to {output_file_path}")

    except Exception as e:
        logging.error(f"An error occurred during the integration test: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Deepgram and ElevenLabs integration.")
    parser.add_argument("input_file", type=str, help="Path to the input MP3 audio file.")
    args = parser.parse_args()

    asyncio.run(main(args.input_file))
