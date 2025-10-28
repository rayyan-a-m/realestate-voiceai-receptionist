# AI Voice Agent for Real Estate

This project is a sophisticated, real-time AI voice agent designed to act as an appointment setter for a real estate firm. It can handle both inbound and outbound calls, qualify leads, provide information on properties, and book site visits directly on a Google Calendar.

## Core Features

- **Real-Time Conversation**: Utilizes a low-latency stack (Deepgram, ElevenLabs, Twilio) for natural, real-time voice interaction.
- **Inbound & Outbound Calls**: Can receive calls from interested clients and proactively call leads from a list.
- **Intelligent Appointment Booking**: The agent checks a Google Calendar for availability in real-time and books appointments using LangChain tools.
- **Context-Aware**: The agent is aware of the specific real estate properties the firm offers and can discuss them with leads.
- **24/7 Availability**: The agent can book appointments around the clock, finding the next available slots.
- **Modular & Scalable**: Built with FastAPI, the system is designed to be scalable and allows for easy integration with other lead sources like CRMs.

## Architecture

The agent uses a combination of best-in-class services orchestrated by a Python FastAPI server.

- **Telephony**: Twilio for handling phone calls (making, receiving, and managing audio streams).
- **Speech-to-Text (STT)**: Deepgram for fast and accurate real-time transcription.
- **Text-to-Speech (TTS)**: ElevenLabs for high-quality, low-latency voice generation.
- **The "Brain"**: A LangChain agent powered by Google's Gemini 1.5 Flash, equipped with tools to interact with Google Calendar.
- **Web Server**: FastAPI for handling webhooks from Twilio and managing WebSocket connections for audio streaming.
- **Calendar**: Google Calendar API for checking availability and creating events.

## Project Structure

```
/voiceAI-agent
|-- .env                  # Stores all API keys and secrets
|-- requirements.txt      # Python dependencies
|-- config.py             # Loads configuration and property details
|-- prompts.py            # Contains the core system prompt for the AI agent
|-- google_calendar.py    # LangChain tools for Google Calendar integration
|-- main.py               # The main FastAPI application logic
|-- credentials.json      # Google API credentials (you must provide this)
|-- token.json            # Google API token (generated on first run)
|-- README.md             # This file
```

## Setup and Installation

### 1. Prerequisites

- Python 3.9+
- A Twilio account with a phone number.
- API keys for:
  - Google AI (Gemini)
  - Deepgram
  - ElevenLabs
- A Google Cloud Platform project with the Google Calendar API enabled.

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd voiceAI-agent
```

### 3. Install Dependencies

Create a virtual environment and install the required packages.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of the project and fill it with your credentials.

```
GOOGLE_API_KEY="AIza..."  # Get from https://aistudio.google.com/app/apikey
DEEPGRAM_API_KEY="..."
ELEVENLABS_API_KEY="..."

# Twilio credentials
TWILIO_ACCOUNT_SID="AC..."
TWILIO_AUTH_TOKEN="..."
TWILIO_PHONE_NUMBER="+1..." # Your Twilio phone number

# Public URL of your deployed server (e.g., from Render)
AGENT_HOST_URL="https://your-voice-agent.onrender.com"

# Business details
YOUR_BUSINESS_NAME="Prestige Properties"
```

**Getting Your Google API Key**:
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and paste it in your `.env` file

**IMPORTANT**: `AGENT_HOST_URL` must be a publicly accessible URL. You cannot use `localhost`. Services like [Render](https://render.com/) or [Railway](https://railway.app/) are excellent for deploying the application.

### 5. Set Up Twilio

Twilio provides the phone number and telephony infrastructure for your voice agent.

#### Step-by-Step Twilio Setup:

1.  **Create a Twilio Account**:
    - Go to [Twilio](https://www.twilio.com/try-twilio) and sign up for a free account.
    - You'll receive some free credits to get started.

2.  **Get Your Account SID and Auth Token**:
    - After logging in, go to your [Twilio Console Dashboard](https://console.twilio.com/).
    - You'll see your **Account SID** and **Auth Token** displayed prominently.
    - Copy these values and add them to your `.env` file:
      ```
      TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      TWILIO_AUTH_TOKEN="your_auth_token_here"
      ```

3.  **Buy a Phone Number**:
    - In the Twilio Console, navigate to **Phone Numbers** → **Manage** → **Buy a number**.
    - Select your country and choose a phone number with **Voice** capabilities.
    - For the free trial, you can get one phone number for free.
    - Once purchased, copy the phone number in E.164 format (e.g., `+14155552671`).
    - Add it to your `.env` file:
      ```
      TWILIO_PHONE_NUMBER="+14155552671"
      ```

4.  **Verify Phone Numbers (Trial Account)**:
    - If you're using a trial account, Twilio requires you to verify any phone numbers you want to call.
    - Go to **Phone Numbers** → **Manage** → **Verified Caller IDs**.
    - Click **Add a new Caller ID** and follow the verification process.
    - Once verified, you can make outbound calls to that number.

5.  **Upgrade Your Account (Optional)**:
    - To remove trial limitations (like verified caller IDs), you can upgrade your account.
    - Go to **Billing** in the Twilio Console and add payment information.

**Note**: After deploying your server, you'll need to configure the webhook URL in Twilio (covered in the Deployment section below).

### 6. Set Up Google Calendar API

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a new project.
3.  Enable the **Google Calendar API**.
4.  Go to "Credentials", click "Create Credentials" -> "OAuth client ID".
5.  Select "Desktop app" as the application type.
6.  Click "Download JSON" and save the file as `credentials.json` in the project root directory.

### 7. Authenticate with Google

Before starting the server, you must run the `google_calendar.py` script once locally to authorize the application and generate a `token.json` file.

```bash
python google_calendar.py
```

This will open a browser window asking you to log in with your Google account and grant permission.

## Deployment and Execution

### Why Render for Prototyping?

For this prototype, a Platform-as-a-Service (PaaS) like **Render** is recommended primarily for its **simplicity and speed**. The goal is to get a functional version of the agent running on a public URL as quickly as possible with minimal configuration.

-   **Ease of Use**: Render offers a straightforward Git-push-to-deploy workflow. You connect your GitHub repository, define your start command, and the platform handles the rest—provisioning servers, installing dependencies, and setting up networking.
-   **Integrated Features**: It provides essential features like managed TLS certificates (for `https`), environment variable management, and auto-deploys out of the box.
-   **Developer Experience**: For a prototype, this simplicity allows you to focus on the application logic rather than on complex cloud infrastructure setup.

### When to Choose GCP (Google Cloud Platform)

**GCP** is an excellent, enterprise-grade choice for scaling this application into a full-fledged product. While it offers more power and flexibility, it also involves a steeper learning curve.

-   **Scalability & Control**: GCP services like Google Kubernetes Engine (GKE) or Cloud Run provide fine-grained control over scaling, networking, and security, which is essential for a production application handling high call volumes.
-   **Ecosystem Integration**: If you are already using other Google Cloud services (like Firestore, Cloud SQL, or Pub/Sub), deploying on GCP makes integration seamless.
-   **Complexity**: Setting up a comparable deployment on GCP would typically require more explicit configuration of services like VPC (Virtual Private Cloud), IAM (Identity and Access Management) permissions, load balancers, and container registries.

**Conclusion**: Start with Render to validate the prototype quickly. Migrate to a more robust platform like GCP when you are ready to scale and require more granular control over your infrastructure.

### 1. Deploy the Server

1.  Sign up for a cloud hosting service like **Render**.
2.  Create a new "Web Service" and connect it to your GitHub repository.
3.  Set the **Start Command** to: `uvicorn main:app --host 0.0.0.0 --port 8000`.
4.  In the "Environment" section of your Render service, add all the key-value pairs from your local `.env` file.
5.  Deploy the service. Render will provide you with a public URL (e.g., `https://your-voice-agent.onrender.com`). Make sure this URL is correctly set as `AGENT_HOST_URL` in your environment variables on Render.

### 2. Configure Twilio Webhook

1.  Go to your Twilio account and navigate to the settings for your phone number.
2.  Under "Voice & Fax", find the "A CALL COMES IN" section.
3.  Set it to "Webhook", and enter your server's public URL followed by `/inbound_call`.
    - Example: `https://your-voice-agent.onrender.com/inbound_call`
4.  Set the HTTP method to `POST`.
5.  Save the configuration.

### 3. Running the Agent

-   **Inbound Calls**: Simply call your Twilio phone number. You will be connected to the AI agent. Check your server logs on Render to see the real-time transcription and agent activity.

-   **Outbound Campaigns**:
    1.  Create a CSV file (e.g., `leads.csv`) with the headers `first_name`, `last_name`, and `phone`. Phone numbers must be in E.164 format (e.g., `+14155552671`).
    2.  Use a tool like `curl` to upload the file to your running server and start the campaign:
        ```bash
        curl -X POST -F "file=@/path/to/your/leads.csv" https://your-voice-agent.onrender.com/start_outbound_campaign
        ```
    3.  The server will begin calling the leads in the file one by one.

## How to Productionize This Prototype

This prototype is a strong foundation, but for a robust, production-ready system, consider the following enhancements:

1.  **State Management**: The current `conversation_history` is stored in-memory, which is not scalable and is lost on server restarts.
    -   **Solution**: Replace the in-memory dictionary with a **Redis** database. Use the `call_sid` as the key to store and retrieve conversation history for each call. This ensures state persistence and scalability.

2.  **Secure Webhooks**: Twilio webhooks are public endpoints. You must verify that incoming requests are genuinely from Twilio.
    -   **Solution**: Use Twilio's `RequestValidator` in a FastAPI middleware. This validator uses your Twilio Auth Token to check the `X-Twilio-Signature` header on incoming requests. Reject any request that fails validation.

3.  **Robust WebSocket Handling**: The prototype uses a simplified method for identifying calls in the WebSocket connection.
    -   **Solution**: Pass the unique `call_sid` in the WebSocket URL itself. Modify the Twilio TwiML response to generate a dynamic URL for each call, like `wss://your-server.com/ws/{call_sid}`. This makes connection management stateless and reliable.

4.  **Observability and Monitoring**: In production, you need to know what's happening.
    -   **Solution**: Integrate a logging and monitoring service like **Datadog**, **Sentry**, or **Grafana**.
        -   **Logging**: Log key events, agent decisions, tool usage, and errors.
        -   **Monitoring**: Track critical metrics like call duration, transcription latency, TTS latency, and agent response time. Set up alerts for high error rates or performance degradation.

5.  **Scalable Lead Management**: The CSV upload is good for demos but not for production.
    -   **Solution**: Abstract the lead source. Create a `LeadProvider` base class and implement different providers:
        -   `CSVLeadProvider`: The current implementation.
        -   `SalesforceLeadProvider`: Connects to the Salesforce API to pull leads from a campaign or report.
        -   `HubSpotLeadProvider`: Does the same for HubSpot.
        Your API endpoint can then take a `source` parameter to choose the lead provider, making the system highly modular.

6.  **Managed Infrastructure (The Fast Path)**: Building and maintaining real-time voice infrastructure is complex. For faster time-to-market and higher reliability, consider using a managed platform.
    -   **Solution**: Use services like **Vapi.ai** or **Bland.ai**. These platforms handle the entire telephony, STT, and TTS stack. You simply provide the LangChain agent logic (the "brain"). This abstracts away the complexity of managing WebSockets, audio encoding, and latency, allowing you to focus purely on the agent's intelligence. Having built this prototype, you are now in an excellent position to use these platforms effectively.
