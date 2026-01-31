import os
import json
import base64
import asyncio
import edge_tts
from groq import Groq
from http.server import BaseHTTPRequestHandler
from duckduckgo_search import DDGS
from datetime import datetime

# HELPER: Async function to generate audio using Edge-TTS
async def generate_audio(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

# SEARCH: Logic to get fresh data
def get_web_results(query):
    print(f"Searching web for: {query}")
    results_text = ""
    try:
        with DDGS() as ddgs:
            # If asking for weather or news, use the News tab for much fresher data
            if any(word in query.lower() for word in ["weather", "news", "price", "stock"]):
                print("Using News Search for fresh data...")
                results = list(ddgs.news(query, max_results=5))
            else:
                results = list(ddgs.text(query, max_results=5))
            
            if results:
                for r in results:
                    # News results use 'body', Text results use 'body' or 'snippet'
                    content = r.get('body', r.get('snippet', ''))
                    results_text += f"SOURCE: {r.get('title')}\nINFO: {content}\n\n"
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
            user_msg_content = messages[-1]['content'] if messages else request_body.get('message', "")

            api_key = os.environ.get("GROQ_API_KEY")
            client = Groq(api_key=api_key)

            # Current context
            current_date = datetime.now().strftime("%A, %b %d, %Y")
            
            # 1. GET FRESH DATA
            search_context = get_web_results(user_msg_content)
            
            # 2. STRICT SYSTEM PROMPT
            system_content = (
                f"Today is {current_date}. You are a factual assistant. "
                "You are provided with search results from the last few hours. "
                "RULE 1: Only report numbers (like temperature) if they are explicitly in the search results. "
                "RULE 2: If the results show multiple temperatures, pick the one from the most recent-looking source. "
                "RULE 3: If you cannot find a specific current temperature, say 'The search results show [General Info] but don't state the exact current temperature.' "
                "Keep response to 1-2 sentences."
            )
            
            if search_context:
                system_content += f"\n\nFRESH SEARCH RESULTS:\n{search_context}"

            system_prompt = {"role": "system", "content": system_content}
            
            # Ensure we send the full message history if it exists
            final_messages = [system_prompt] + (messages if messages else [{"role": "user", "content": user_msg_content}])

            # 3. Generate Text
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=final_messages,
                temperature=0.0, # Zero temperature for absolute factual rigidity
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
