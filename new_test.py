import asyncio
import base64
import json
import wave
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_test(uri):
    """Connects to the WebSocket, sends a start message, and streams audio."""
    try:
        async with websockets.connect(uri) as websocket:
            logging.info("WebSocket connection established.")
            
            # 1. Send 'connected' and 'start' messages to initiate the call
            await websocket.send(json.dumps({"event": "connected"}))
            start_message = {
                "event": "start",
                "start": {
                    "callSid": "test_call_sid",
                    "streamSid": "test_stream_sid"
                }
            }
            await websocket.send(json.dumps(start_message))
            logging.info("Sent start message.")

            # 2. Simulate initial user speech (e.g., a greeting)
            logging.info("Simulating initial user speech...")
            # This can be a short, silent audio chunk just to trigger the agent's response
            silent_chunk = b'\x00' * 1600
            media_message = {
                "event": "media",
                "streamSid": "test_stream_sid",
                "media": {
                    "payload": base64.b64encode(silent_chunk).decode('utf-8')
                }
            }
            await websocket.send(json.dumps(media_message))
            
            # Give the agent time to respond
            await asyncio.sleep(5) 

            # 3. Simulate an interruption while the agent is likely speaking
            logging.info("Simulating user interruption...")
            # Send another audio chunk to trigger the interruption logic
            await websocket.send(json.dumps(media_message))
            
            logging.info("Interruption sent. Monitor server logs to verify.")
            
            # Keep the connection open for a bit longer to observe server behavior
            await asyncio.sleep(10)

            # 4. Stop the call
            stop_message = {
                "event": "stop",
                "stop": {
                    "callSid": "test_call_sid"
                }
            }
            await websocket.send(json.dumps(stop_message))
            logging.info("Sent stop message.")

    except Exception as e:
        logging.error(f"Test failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure your FastAPI server is running before executing this test
    # The default URL is for a local server running on port 8000
    websocket_uri = "ws://localhost:8000/ws"
    asyncio.run(run_test(websocket_uri))
