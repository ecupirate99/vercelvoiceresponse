import os
import json
import base64
import asyncio
import edge_tts
from groq import Groq
from http.server import BaseHTTPRequestHandler
from duckduckgo_search import DDGS

# HELPER: Async function to generate audio using Edge-TTS
async def generate_audio(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

# IMPROVED: Search function
def get_web_results(query):
    print(f"Searching web for: {query}")
    try:
        results_text = ""
        with DDGS() as ddgs:
            # We increased max_results to 5 to get a better chance of finding the data
            results = list(ddgs.text(query, max_results=5))
            if results:
                for r in results:
                    results_text += f"- {r['body']}\n"
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
            if content_length == 0:
                self.send_error_response(400, "Request body is empty.")
                return

            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            
            selected_voice = request_body.get('voice', 'en-US-AriaNeural')
            messages = request_body.get('messages', [])
            
            user_msg_content = ""
            if not messages:
                user_msg_content = request_body.get('message', "")
                messages = [{"role": "user", "content": user_msg_content}]
            else:
                user_msg_content = messages[-1]['content']

            api_key = os.environ.get("GROQ_API_KEY")
            client = Groq(api_key=api_key)

            # 1. SEARCH THE WEB
            search_context = get_web_results(user_msg_content)
            
            # 2. IMPROVED SYSTEM PROMPT
            # We are now telling the AI it HAS access to the web via these results.
            system_content = (
                "You are a helpful assistant with real-time web access. "
                "Use the provided web search results to answer the user's question accurately. "
                "If the search results contain the answer (like weather or news), state it clearly. "
                "Keep your response very concise (1-2 sentences)."
            )
            
            if search_context:
                system_content += f"\n\nCURRENT WEB SEARCH RESULTS:\n{search_context}"

            system_prompt = {"role": "system", "content": system_content}
            final_messages = [system_prompt] + messages

            # 3. Generate Text
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=final_messages,
                temperature=0.3, # Lowered temperature for more factual accuracy
                max_tokens=200, 
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
