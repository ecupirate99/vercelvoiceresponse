# Voice Chat Bot

A real-time Voice Chat Bot application optimized for deployment on **Vercel** with a Python backend and static frontend. 

The bot allows users to talk or type messages, utilizes a web search integration to answer factual queries with real-time data, and responds back in voice using high-quality neural Text-To-Speech (TTS).

---

## 🚀 Features

- **Microphone Integration (Speech-to-Text)**: An interactive, pulsing mic button near the chat input allows users to dictate messages. Once speech ends, it auto-transcribes and submits.
- **Edge TTS Integration (Text-to-Speech)**: Converts the LLM's response into high-quality neural voice audio dynamically, with a voice dropdown selector (Aria, Guy, Jenny, Sonia, Christopher).
- **LLM + Web Search Context**: Integrates with DuckDuckGo Search to fetch fresh factual data for queries (like current weather, news, or stats) and formats the context into a prompt for Groq (`llama-3.1-8b-instant`).
- **Responsive Premium UI**: Glassmorphism design elements, clean loading animations, and smooth chat history scrolling.

---

## 🛠️ Local Development & Testing

You can test the application locally using the provided multi-threaded dev server (`local_server.py`), which maps the frontend files and serverless API endpoints.

### Prerequisites
Make sure you have Python installed.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Optionally install `python-dotenv` if not already installed).*

2. **Configure your API Key**:
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Start the Local Server**:
   ```bash
   python local_server.py
   ```
   *For Windows environments using Python Launcher:*
   ```bash
   py local_server.py
   ```

4. **Access the App**:
   Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## ☁️ Deployment to Vercel

This repository is pre-configured for Vercel deployment:

1. Connect this repository to your Vercel project.
2. Add your **`GROQ_API_KEY`** to the **Environment Variables** in your Vercel Project Dashboard.
3. Deploy! Vercel will automatically route the frontend assets via `public/index.html` and compile the Python function under `api/chat.py` as a serverless endpoint.