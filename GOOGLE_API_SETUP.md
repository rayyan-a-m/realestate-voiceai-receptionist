# Google Cloud API & Credentials Setup Guide

This guide provides detailed steps to set up the necessary Google Cloud APIs and credentials for your real-time voice AI agent.

---

## 📋 Prerequisites

- A Google Account with billing enabled.
- The `gcloud` command-line tool installed and authenticated (`gcloud auth login`).

---

## Step 1: Create or Select a Google Cloud Project

Every Google Cloud resource belongs to a project.

1.  **Go to the Google Cloud Console:** [https://console.cloud.google.com/](https://console.cloud.google.com/)
2.  In the top bar, click the project selector dropdown.
3.  Either select an existing project or click **"NEW PROJECT"**.
4.  Give your project a name (e.g., `voice-ai-agent`) and click **"CREATE"**.
5.  Make sure your new project is selected in the dropdown.

---

## Step 2: Enable Required APIs

You need to enable three core APIs for this project.

1.  **Go to the API Library:** In the Cloud Console, navigate to **APIs & Services > Library**.
2.  **Enable Vertex AI API:**
    -   Search for "Vertex AI API".
    -   Click on it and then click **"ENABLE"**.
3.  **Enable Cloud Speech-to-Text API:**
    -   Search for "Cloud Speech-to-Text API".
    -   Click on it and then click **"ENABLE"**.
4.  **Enable Cloud Text-to-Speech API:**
    -   Search for "Cloud Text-to-Speech API".
    -   Click on it and then click **"ENABLE"**.

Wait for a few moments for the APIs to be fully enabled in your project.

---

## Step 3: Create a Service Account

A service account is a special type of Google account that belongs to your application instead of an individual user. Your application will use this account to authenticate with Google Cloud services.

1.  **Go to Service Accounts:** In the Cloud Console, navigate to **IAM & Admin > Service Accounts**.
2.  Click **"+ CREATE SERVICE ACCOUNT"**.
3.  **Service account details:**
    -   **Service account name:** `voice-ai-service-account` (or another descriptive name).
    -   **Service account ID:** This will be automatically generated.
    -   **Description:** "Service account for the real-time voice AI agent".
4.  Click **"CREATE AND CONTINUE"**.
5.  **Grant permissions:**
    -   Click the **"Select a role"** dropdown.
    -   Search for and add the following roles one by one:
        -   `Vertex AI User`: Allows access to Vertex AI models like Gemini.
        -   `Cloud Speech-to-Text User`: Allows use of the STT API.
        -   `Cloud Text-to-Speech User`: Allows use of the TTS API.
    -   *Note: For production, it's best practice to create custom roles with the minimum required permissions.*
6.  Click **"CONTINUE"**.
7.  Click **"DONE"** (you can skip granting users access to this service account).

---

## Step 4: Create and Download a Service Account Key

The key is a JSON file that contains the private credentials for your service account. Treat this file like a password—do not commit it to version control.

1.  **Find your new service account** in the list.
2.  Click the three-dots menu (Actions) on the right side and select **"Manage keys"**.
3.  Click **"ADD KEY" > "Create new key"**.
4.  Select **JSON** as the key type and click **"CREATE"**.
5.  A JSON file will be automatically downloaded to your computer. It will be named something like `[PROJECT_ID]-[UNIQUE_ID].json`.
6.  **Rename and move the file:**
    -   Rename the file to something simple, like `credentials.json`.
    -   Move it to a secure location within your project directory. **Crucially, add this filename to your `.gitignore` file** to prevent it from being committed to Git.

    ```
    # .gitignore
    credentials.json
    __pycache__/
    *.env
    ```

---

## Step 5: Set the Environment Variable

Your application code will look for an environment variable named `GOOGLE_APPLICATION_CREDENTIALS` to find the key file and authenticate.

You can set this variable in your terminal for the current session:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

**Replace `/path/to/your/credentials.json` with the absolute path to the JSON file you downloaded.**

To make this permanent, you can add the `export` command to your shell's startup file (e.g., `~/.zshrc`, `~/.bashrc`) and then run `source ~/.zshrc` or open a new terminal.

Alternatively, if you are using a `.env` file with `python-dotenv`, you can add it there:

```
# .env file
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

Your application is now configured to securely authenticate with Google Cloud!
