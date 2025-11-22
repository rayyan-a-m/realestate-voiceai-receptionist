import asyncio
import uuid
import logging

# Configure basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NOTE: This script targeted the legacy LangChain agent that lived inside main.py.
# After migrating to the Gemini Live pipeline there is no longer an in-process
# agent object to invoke directly, so the previous test harness is not
# applicable. We surface this early to avoid confusing ImportErrors later on.
raise SystemExit(
    "test_agent.py has been deprecated now that Gemini Live handles voice calls. "
    "Use the live Twilio media stream (or local_test.py) to exercise the agent."
)

async def run_agent_test():
    """
    Tests the LangChain agent by invoking it with a sample conversation
    that should trigger tool calls for finding and booking appointments.
    """
    # A unique session ID for this test conversation, simulating a single call.
    session_id = f"test-session-{uuid.uuid4()}"
    
    print("\n--- Starting Agent Test ---")
    print(f"Using Session ID: {session_id}")

    # --- Test Case 1: A simple greeting to start the conversation ---
    print("\n[Test 1: Greeting]")
    input_1 = "Hi there, my name is Jane Doe."
    print(f"User > {input_1}")
    
    # The .ainvoke method runs the agent and returns the final result
    response_1 = await agent_with_chat_history.ainvoke(
        {"input": input_1},
        config={"configurable": {"session_id": session_id}},
    )
    
    agent_output_1 = response_1.get('output', 'No output found.')
    print(f"Agent > {agent_output_1}")
    print("-" * 30)

    # --- Test Case 2: A query that should trigger the 'find_available_slots' tool ---
    print("\n[Test 2: Tool Call - Find Availability]")
    # Note: The date is hardcoded for predictability. The tool should find slots for Nov 16, 2025.
    input_2 = "I'm interested in viewing a property. Could you check for any available times tomorrow?"
    print(f"User > {input_2}")

    response_2 = await agent_with_chat_history.ainvoke(
        {"input": input_2},
        config={"configurable": {"session_id": session_id}},
    )
    
    agent_output_2 = response_2.get('output', 'No output found.')
    print(f"Agent > {agent_output_2}")
    print("-" * 30)

    # --- Test Case 3: A follow-up to book a specific time, triggering 'book_appointment' ---
    # This test assumes the previous step successfully found and returned available slots.
    # We'll choose a plausible time from the expected output of the find_available_slots tool.
    print("\n[Test 3: Tool Call - Book Appointment]")
    input_3 = "Perfect. Please book the 10:00 AM slot for me. My email is jane.doe@example.com and my phone number is 555-123-4567."
    print(f"User > {input_3}")

    response_3 = await agent_with_chat_history.ainvoke(
        {"input": input_3},
        config={"configurable": {"session_id": session_id}},
    )

    agent_output_3 = response_3.get('output', 'No output found.')
    print(f"Agent > {agent_output_3}")
    print("-" * 30)

    # --- Verification (Optional) ---
    # You can inspect the chat history to see the full conversation flow
    print("\n[Verification: Final Chat History]")
    history = session_histories.get(session_id)
    if history:
        for message in history.messages:
            print(f"- {message.type.upper()}: {message.content}")
    else:
        print("Chat history not found.")

    print("\n--- Agent Test Finished ---")


if __name__ == "__main__":
    # This script requires a running event loop to execute the async agent calls.
    # Before running, ensure you have authenticated with Google Cloud for Vertex AI and Google Calendar.
    # You may need to run `gcloud auth application-default login` in your terminal.
    # Also, the first run of `google_calendar.py` (or this script) might trigger a browser-based
    # OAuth flow to get a `token.json` for calendar access.
    try:
        asyncio.run(run_agent_test())
    except Exception as e:
        logging.error(f"An error occurred during the test run: {e}", exc_info=True)
