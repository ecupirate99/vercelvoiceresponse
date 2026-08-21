# vercelvoiceresponse

A small voice-enabled assistant deployed on Vercel that:

- Sends user messages to a Groq LLM (model: `openai/gpt-oss-20b`) to produce concise, factual replies.
- Uses duckduckgo-search to fetch very recent web results and inject them into a strict system prompt for up-to-the-minute context.
- Generates spoken responses with Edge TTS (`edge-tts`).
- Simple web UI (public/index.html and public/index2.html) with voice selection and an audio player for replies.

---

## Latest updates (summary)

- Server: api/chat.py
  - Uses the Groq client (requires `GROQ_API_KEY`) to call `openai/gpt-oss-20b` for chat completions.
  - Adds a strict system prompt that includes fresh search results from DuckDuckGo when available.
  - Produces TTS audio using `edge-tts` (async streaming) and returns base64-encoded WAV audio alongside the text reply.

- Client: public/index.html and public/index2.html
  - Minimal chat interface that POSTs to `/api/chat` and, if `audio` is returned, plays it with an HTML audio element.
  - Voice selector that sends the chosen voice name to the server (`voice` field).

- Requirements (see requirements.txt):
  - edge-tts
  - groq
  - duckduckgo-search

---

## How it works

1. Client POSTs message (and optional `voice`) to `/api/chat`.
2. Server builds a system prompt with the current date and recent DuckDuckGo search results, then calls Groq chat completions with `openai/gpt-oss-20b`.
3. Server synthesizes the response text into audio via `edge-tts` and returns JSON: `{ text: string, audio: base64? }`.

---

## Environment variables

- `GROQ_API_KEY` (required) — API key used by the Groq client to call the LLM.

Note: There is no `OPENAI_API_KEY` or separate STT key in this repo by default — transcription (speech-to-text) is not implemented on the server in the current code.

---

## Running locally (quick)

1. Create a Python virtualenv and install requirements:

   pip install -r requirements.txt

2. Set your environment variable:

   export GROQ_API_KEY="your_groq_api_key"

3. Start the Python server entrypoint (the repo contains a minimal http.server-based handler in `api/chat.py` — adapt to your environment or use a small WSGI wrapper if necessary).

4. Serve `public/` (e.g., with a static file server or deploy to Vercel) so the web UI can reach your `/api/chat` endpoint.

---

## Known limitations & troubleshooting (mic / transcription)

Symptom: When clicking the microphone icon in your web UI you see "could not transcribe audio".

Based on the repository contents, that error is NOT caused by the LLM (the Groq model `openai/gpt-oss-20b`) because the server currently only handles generating replies and TTS; there is no server-side speech-to-text implementation in this repo. Here are the likely causes and steps to debug:

1. Missing transcription endpoint
   - The client may be trying to POST recorded audio to a `/api/transcribe` (or similar) endpoint that does not exist in this repo. Search the client JavaScript (browser console / network tab) for the request the mic button triggers.

2. Client-side failure (MediaRecorder / permissions)
   - Ensure the browser granted microphone permission.
   - Check the browser console for errors from `navigator.mediaDevices.getUserMedia` or `MediaRecorder`.

3. External STT provider / model name changes
   - If you previously used Whisper (OpenAI) or a Groq speech-to-text model, confirm that the server code actually calls that provider and that the provider API key and model name are current.
   - Model name changes (e.g., provider renaming their STT model) only matter if you have server code that calls that model. This repo does not include such server-side transcription code, so a model rename in the LLM alone would not produce the UI message.

4. How to add working transcription
   - Add a server endpoint (e.g., `/api/transcribe`) that accepts recorded audio (webm/wav), and forward it to a speech-to-text API (OpenAI Whisper, Deepgram, AssemblyAI, Groq STT if available) using the provider's current model name and API key. Return the transcribed text to the client.
   - Alternatively, you can use the browser SpeechRecognition API for basic live transcribe (note: browser support and accuracy vary).

5. Quick checks to run now
   - Open DevTools → Console & Network, perform the mic flow, and capture the failing request and any console stack trace.
   - If you see a 404 to `/api/transcribe` or similar, that's the missing server endpoint.
   - If you see a 403/401 from a provider endpoint, check your transcription provider API key and model name.

---

## Suggested next changes

- Add a `/api/transcribe` handler that accepts audio and uses a speech-to-text API. Example providers:
  - OpenAI Whisper (`openai/whisper-1`) — requires an OpenAI API key and the proper request format.
  - Groq STT (if they provide an STT model) — check Groq docs for the current model ID and request pattern.
  - Browser SpeechRecognition for an all-client solution (no server costs, but lower accuracy and limited language support).

- Add clearer UI error messages to surface network errors or missing server endpoints.

---

If you'd like, I can:

- Add a sample `/api/transcribe` endpoint (server-side example calling OpenAI Whisper or a Groq STT) plus client code to record and upload audio, or
- Update README further with step-by-step deployment instructions for Vercel.

Tell me which option you prefer and I'll add it to the repo.