import os
import json
import base64
import asyncio
import edge_tts
from groq import Groq
from http.server import BaseHTTPRequestHandler
# NEW: Import DuckDuckGo for free web search
from duckduckgo_search import DDGS

# HELPER: Async function to generate audio using Edge-TTS
async def generate_audio(text, voice):
    # Use the voice passed from the function call
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
            
    return audio_data

# NEW: Function to search the web
def get_web_results(query):
    print(f"Searching web for: {query}")
    try:
        results_text = ""
        # Search for top 3 results
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                for r in results:
                    results_text += f"Title: {r['title']}\nSnippet: {r['body']}\n\n"
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
            # 1. Validation & Setup
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, "Request body is empty.")
                return

            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            
            # Get voice and message
            selected_voice = request_body.get('voice', 'en-US-AriaNeural')
            messages = request_body.get('messages', [])
            
            # Fallback for simple testing
            user_msg_content = ""
            if not messages:
                user_msg_content = request_body.get('message')
                if user_msg_content:
                    messages = [{"role": "user", "content": user_msg_content}]
                else:
                    self.send_error_response(400, "No 'messages' provided.")
                    return
            else:
                # Grab the last user message to use as the search query
                user_msg_content = messages[-1]['content']

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                self.send_error_response(500, "GROQ_API_KEY missing.")
                return
            
            client = Groq(api_key=api_key)

            # 2. WEB SEARCH INTEGRATION
            # We search based on the user's latest message
            search_context = get_web_results(user_msg_content)
            
            system_content = "You are a helpful assistant. Keep your answers very concise, short, and to the point. Limit responses to 1-2 sentences."
            
            # If we found web results, add them to the system prompt
            if search_context:
                system_content += f"\n\nHERE IS REAL-TIME WEB INFO. USE THIS TO ANSWER:\n{search_context}"

            system_prompt = {
                "role": "system", 
                "content": system_content
            }
            
            final_messages = [system_prompt] + messages

            # 3. Generate Text (Llama 3)
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=final_messages,
                temperature=0.7,
                max_tokens=200, 
            )
            response_text = chat_completion.choices[0].message.content

            # 4. Generate Audio (Edge TTS)
            try:
                audio_bytes = asyncio.run(generate_audio(response_text, selected_voice))
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception as e:
                print(f"Audio generation failed: {e}")
                audio_base64 = None

            # 5. Send Response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "text": response_text,
                "audio": audio_base64 
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error_response(500, str(e))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
