import os
import json
import base64
import asyncio
import edge_tts
from groq import Groq
from http.server import BaseHTTPRequestHandler
from duckduckgo_search import DDGS
from datetime import datetime # Added to track current time

# HELPER: Async function to generate audio using Edge-TTS
async def generate_audio(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

# SEARCH: Optimized to find current info
def get_web_results(query):
    # Add "current" and "today" to the query to force newer results
    search_query = f"{query} current weather temperature today" if "weather" in query.lower() else query
    print(f"Searching web for: {search_query}")
    
    try:
        results_text = ""
        with DDGS() as ddgs:
            # We look at more results to find the most recent one
            results = list(ddgs.text(search_query, max_results=6))
            if results:
                for r in results:
                    results_text += f"- Source: {r['title']} | Info: {r['body']}\n"
                return results_text
    except Exception as e:
        print(f"Search error: {e}")
    return None

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            
            selected_voice = request_body.get('voice', 'en-US-AriaNeural')
            messages = request_body.get('messages', [])
            
            if not messages:
                user_msg_content = request_body.get('message', "")
                messages = [{"role": "user", "content": user_msg_content}]
            else:
                user_msg_content = messages[-1]['content']

            api_key = os.environ.get("GROQ_API_KEY")
            client = Groq(api_key=api_key)

            # Get current date/time to help the AI filter old search results
            current_time = datetime.now().strftime("%A, %B %d, %Y")

            # 1. SEARCH THE WEB
            search_context = get_web_results(user_msg_content)
            
            # 2. STRICT SYSTEM PROMPT
            system_content = (
                f"Today's date is {current_time}. "
                "You are a factual assistant with real-time web access. "
                "Use the search results provided to give accurate, current information. "
                "Ignore results that appear to be from past years or look like generic climate data. "
                "If the search results mention a specific temperature for today, use that. "
                "Keep your response to 1-2 sentences max."
            )
            
            if search_context:
                system_content += f"\n\nWEB SEARCH RESULTS:\n{search_context}"

            system_prompt = {"role": "system", "content": system_content}
            final_messages = [system_prompt] + messages

            # 3. Generate Text (Lower temperature = more factual)
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=final_messages,
                temperature=0.1, 
                max_tokens=150, 
            )
            response_text = chat_completion.choices[0].message.content

            # 4. Generate Audio
            try:
                audio_bytes = asyncio.run(generate_audio(response_text, selected_voice))
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception:
                audio_base64 = None

            # 5. Send Response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {"text": response_text, "audio": audio_base64}
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_error_response(500, str(e))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
