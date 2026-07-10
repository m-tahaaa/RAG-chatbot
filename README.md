# PDF RAG Backend

A simple FastAPI backend with a static chatbot UI that answers only from one PDF already stored in Azure Blob Storage.

## What it does

- loads one configured PDF from Azure Blob Storage
- chunks and indexes it in Azure AI Search
- answers questions with Azure OpenAI / Azure AI Foundry
- returns JSON from `/ask`
- serves a simple browser UI from `/`

## Required environment variables

Set these in `.env` or in your hosting environment:

```env
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX=
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_CONTAINER=
AZURE_STORAGE_BLOB_NAME=
AZURE_SOURCE_DOCUMENT_ID=source-pdf
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2025-08-07
```

Notes:

- `AZURE_STORAGE_BLOB_NAME` must be the exact blob path of the PDF you already uploaded.
- `AZURE_SOURCE_DOCUMENT_ID` is the internal ID used by Search. Leave it as-is unless you want to rename the source document.
- The backend re-indexes that one blob on startup.

## Run locally

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Make sure the dependencies are installed:

```powershell
pip install -r requirements.txt
```

3. Start the backend:

```powershell
uvicorn main:app --reload
```

4. Open the UI in your browser:

```text
http://127.0.0.1:8000/
```

5. Ask questions in the page. The chatbot will answer only from the configured PDF.

## API

- `GET /health`
- `GET /document`
- `POST /ask`

Example request to `/ask`:

```json
{
  "question": "What is the document about?",
  "top_k": 5
}
```

## Deploy on Azure

If you move this to Azure App Service:

- set the same environment variables in App Settings
- point the app to your existing blob PDF
- make sure the search index exists or let the app create it at startup
- deploy the FastAPI app as-is
