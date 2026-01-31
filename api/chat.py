# api/chat.py
import os
import json
from groq import Groq
from http.server import BaseHTTPRequestHandler
import base64
import uuid
import tempfile

# Initialize Groq client
# Vercel automatically loads environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Create a temporary directory to store audio files
# In a real production app, you'd use a more persistent storage like S3
TEMP_DIR = tempfile.mkdtemp()

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Set response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') # Allow requests from any origin
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        # Read the request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_body = json.loads(post_data.decode('utf-8'))
        
        user_message = request_body.get('message', '')

        try:
            # 1. Generate a text response using Groq's LLM
            # Using Llama 3 8B for faster responses on the free tier
            chat_completion = client.chat.completions.create(
                model="llama3-8b-8192", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond in a conversational and friendly manner."},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500, # Reduced for faster processing
            )
            
            response_text = chat_completion.choices[0].message.content

            # 2. Convert the text response to speech
            speech_response = client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="troy",
                input=response_text,
                response_format="wav"
            )
            
            # 3. Encode the audio data in base64 to send it in the JSON response
            audio_data = speech_response.content
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            
            # 4. Return the text and audio data
            response = {
                "text": response_text,
                "audio": audio_base64
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            # Handle errors gracefully
            error_response = {
                "error": str(e),
                "text": "Sorry, I encountered an error. Please try again."
            }
            self.wfile.write(json.dumps(error_response).encode())

    def do_OPTIONS(self):
        # Handle preflight requests for CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
