# api/chat.py
import os
import json
import base64
from groq import Groq
from http.server import BaseHTTPRequestHandler

# A simple in-memory store for conversation history.
# In a real app with multiple users, you'd use a database keyed by user ID.
conversation_history = []

def truncate_conversation(messages, max_tokens=7000):
    """Truncates the conversation history to fit within the model's token limit."""
    if not messages:
        return messages
    
    system_prompt = messages[0]
    other_messages = messages[1:]
    
    truncated_messages = [system_prompt]
    total_length = len(system_prompt['content'])
    
    for msg in reversed(other_messages):
        msg_length = len(msg['content'])
        if total_length + msg_length < max_tokens:
            truncated_messages.insert(1, msg)
            total_length += msg_length
        else:
            break
            
    return truncated_messages


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        global conversation_history
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            # --- DEBUGGING: Print raw headers and content length ---
            print("--- New Request ---")
            print(f"Headers: {self.headers}")
            print(f"Content-Length: {self.headers.get('Content-Length')}")

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            
            client = Groq(api_key=api_key)

            content_length = int(self.headers['Content-Length'])
            
            # --- DEBUGGING: Handle cases where content might be empty ---
            if content_length == 0:
                raise ValueError("Request body is empty.")

            post_data = self.rfile.read(content_length)
            
            # --- DEBUGGING: Print the raw and parsed body ---
            print(f"Raw post data: {post_data}")
            
            request_body = json.loads(post_data.decode('utf-8'))
            print(f"Parsed JSON body: {request_body}")
            
            user_message = request_body.get('message', '')
            print(f"Extracted user message: '{user_message}'")
            
            if not user_message:
                raise ValueError("No 'message' field found in request.")

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_message})

            messages_for_api = [
                {"role": "system", "content": "You are a helpful assistant. Respond in a conversational and friendly manner."}
            ] + conversation_history

            messages_for_api = truncate_conversation(messages_for_api)
            print(f"Messages being sent to Groq: {messages_for_api}")

            # Generate text response
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # <--- THIS IS THE CORRECTED MODEL NAME
                messages=messages_for_api,
                temperature=0.7,
                max_tokens=500,
            )
            response_text = chat_completion.choices[0].message.content

            # Add bot's response to history
            conversation_history.append({"role": "assistant", "content": response_text})

            # Convert text to speech
            speech_response = client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="troy",
                input=response_text,
                response_format="wav"
            )
            
            audio_base64 = base64.b64encode(speech_response.content).decode("utf-8")
            
            response = {
                "text": response_text,
                "audio": audio_base64
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            # --- DEBUGGING: Print the full exception ---
            import traceback
            print(f"An error occurred: {e}")
            print("Full traceback:")
            traceback.print_exc()
            
            error_response = {
                "error": str(e), 
                "text": "Sorry, I had trouble processing that. Please check the server logs for details."
            }
            self.wfile.write(json.dumps(error_response).encode())
