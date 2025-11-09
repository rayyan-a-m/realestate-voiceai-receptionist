import asyncio
import logging
import warnings
from dotenv import load_dotenv
import os
from vertexai import agent_engines
import config

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Quiet known deprecation noise from Vertex AI SDK during test runs
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"vertexai\.generative_models\._generative_models",
)

async def test_llm_connection():
    logging.info("Initializing Vertex AI LangChain Agent...")

    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials or not os.path.exists(credentials):
        logging.error("Google Cloud credentials are not loaded or not found.")
        print("\n--- LLM Connection Test ---")
        print("❌ Test Failed: Google Cloud credentials not found.")
        print("---------------------------\n")
        return

    try:
        agent = agent_engines.LangchainAgent(
            model="gemini-2.5-flash-lite",
            tools=[],  # if you don’t have any tool functions yet
            model_kwargs={
                "temperature": 0.0,
                "max_output_tokens": 256,
                "top_p": 0.95,
            },
        )
        logging.info("LLM Agent initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize LLM Agent: {e}", exc_info=True)
        print("\n--- LLM Connection Test ---")
        print("❌ Test Failed: Could not initialize the LLM Agent.")
        print(f"Error: {e}")
        print("---------------------------\n")
        return

    try:
        logging.info("Querying agent with a test message...")
        response = agent.query(input="Hello, can you hear me? Respond with a simple confirmation.")
        # response could be a string or dict depending on version
        logging.info(f"Agent response received: {response}")
        logging.info(f"LLM Agent response type - {type(response)}.")
        if isinstance(response, dict):
            response_content = response.get("output", "")
        else:
            response_content = str(response)

        print("\n--- LLM Connection Test ---")
        print(f"Response: {response_content}")
        print("---------------------------\n")

        if any(k in response_content.lower() for k in ["yes", "hear", "sure", "loud and clear"]):
            print("✅ Test Passed: LLM responded as expected.")
        else:
            print("⚠️ Test Warning: LLM responded, but output was not in the expected format.")

    except Exception as e:
        logging.error(f"Agent query failed: {e}", exc_info=True)
        print("\n--- LLM Connection Test ---")
        print("❌ Test Failed: Could not get a response from the LLM Agent.")
        print(f"Error: {e}")
        print("---------------------------\n")

if __name__ == "__main__":
    print("Running Vertex AI LangChain Agent connection test...")
    asyncio.run(test_llm_connection())
