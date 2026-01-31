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
    """
    Truncates the conversation history to fit within the model's token limit.
    It keeps the system prompt and the most recent messages.
    """
    if not messages:
        return messages
    
    # Always keep the system prompt
    system_prompt = messages[0]
    other_messages = messages[1:]
    
    # Simple truncation: keep the last N messages
    # This is a basic approach; more sophisticated methods exist.
    truncated_messages = [system_prompt]
    total_length = len(system_prompt['content'])
    
    # Iterate from the end (most recent messages) to the beginning
    for msg in reversed(other_messages):
        msg_length = len(msg['content'])
        if total_length + msg_length < max_tokens:
            truncated_messages.insert(1, msg) # Insert after system prompt
            total_length += msg_length
        else:
            break # Stop if adding this message would exceed the limit
            
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
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            
            client = Groq(api_key=api_key)

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            user_message = request_body.get('message', '')
            
            if not user_message:
                raise ValueError("No message provided in request.")

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_message})

            # Prepare the full message list for the API
            messages_for_api = [
                {"role": "system", "content": "You are a helpful assistant. Respond in a conversational and friendly manner."}
            ] + conversation_history

            # IMPORTANT: Truncate the history to prevent 400 errors
            messages_for_api = truncate_conversation(messages_for_api)

            # Generate text response
            chat_completion = client.chat.completions.create(
                model="llama3-8b-8192", 
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
            # This will catch the 400 error from Groq and any other exceptions
            print(f"An error occurred: {e}") # Check Vercel logs for this message
            # Send a user-friendly error message back to the frontend
            error_response = {
                "error": str(e), 
                "text": "Sorry, I had trouble processing that. Maybe the message was too long? Please try again."
            }
            self.wfile.write(json.dumps(error_response).encode())
