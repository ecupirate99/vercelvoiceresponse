import os
import json
import base64
import asyncio
import edge_tts
from groq import Groq
from http.server import BaseHTTPRequestHandler

# HELPER: Async function to generate audio using Edge-TTS
async def generate_audio(text):
    # Voices: "en-US-AriaNeural" (Female), "en-US-GuyNeural" (Male), "en-GB-SoniaNeural" (British), etc.
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio_data = b""
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
            
    return audio_data

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
            
            # Get messages from frontend
            messages = request_body.get('messages', [])
            
            # Fallback for simple testing
            if not messages:
                user_msg = request_body.get('message')
                if user_msg:
                    messages = [{"role": "user", "content": user_msg}]
                else:
                    self.send_error_response(400, "No 'messages' provided.")
                    return

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                self.send_error_response(500, "GROQ_API_KEY missing.")
                return
            
            client = Groq(api_key=api_key)

            # 2. STRICT System Prompt for Conciseness
            # We explicitly tell it to be short.
            system_prompt = {
                "role": "system", 
                "content": "You are a helpful assistant. Keep your answers very concise, short, and to the point. Limit responses to 1-2 sentences unless asked for more details."
            }
            
            final_messages = [system_prompt] + messages

            # 3. Generate Text (Llama 3)
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=final_messages,
                temperature=0.7,
                max_tokens=150, # Hard limit on tokens to prevent long rants
            )
            response_text = chat_completion.choices[0].message.content

            # 4. Generate Audio (Edge TTS)
            # We use asyncio.run because do_POST is synchronous, but edge-tts is async
            try:
                audio_bytes = asyncio.run(generate_audio(response_text))
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
