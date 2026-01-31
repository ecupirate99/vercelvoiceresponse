import os
import json
from groq import Groq
from http.server import BaseHTTPRequestHandler

# Note: We removed the global conversation_history. 
# State must be managed by the Frontend or a Database.

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            # 1. Read Request Body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, "Request body is empty.")
                return

            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            
            # 2. Get Messages from Frontend (Stateless approach)
            # Expecting structure: { "messages": [ {"role": "user", "content": "hi"} ] }
            messages = request_body.get('messages', [])
            
            if not messages:
                # Fallback if user sends just a single string 'message'
                user_msg = request_body.get('message')
                if user_msg:
                    messages = [{"role": "user", "content": user_msg}]
                else:
                    self.send_error_response(400, "No 'messages' provided.")
                    return

            # 3. Setup Groq Client
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                self.send_error_response(500, "Server misconfiguration: API Key missing.")
                return
            
            client = Groq(api_key=api_key)

            # 4. Prepare System Prompt
            system_prompt = {
                "role": "system", 
                "content": "You are a helpful assistant. Respond in a conversational and friendly manner."
            }
            
            # Combine system prompt with user history
            # (Ensure system prompt is always first)
            final_messages = [system_prompt] + messages

            # 5. Call Groq for Text
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=final_messages,
                temperature=0.7,
                max_tokens=500,
            )
            response_text = chat_completion.choices[0].message.content

            # NOTE: Groq does NOT support TTS (Audio generation) natively yet.
            # If you need Audio, you must use OpenAI, ElevenLabs, or Deepgram here.
            # Returning text only for now to prevent crash.

            # 6. Send Success Response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "text": response_text,
                "audio": None # Placeholder until you add a valid TTS provider
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            import traceback
            print(f"Server Error: {e}")
            traceback.print_exc()
            self.send_error_response(500, str(e))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
