from fastapi import FastAPI, Request
import os
import logging
from google import genai

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def health():
    return {"status": "running"}

# -----------------------------
# GEMINI CLIENT
# -----------------------------
GEMINI_MODEL = "gemini-2.0-flash"

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    return genai.Client(api_key=api_key)

# -----------------------------
# GEMINI CHAT
# -----------------------------
def ask_gemini(question):

    client = get_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question
    )

    return response.text


# -----------------------------
# CHAT API
# -----------------------------
@app.post("/chat")
async def chat(request: Request):

    data = await request.json()
    message = data.get("message")

    if not message:
        return {"reply": "Please enter a message"}

    try:
        reply = ask_gemini(message)
        return {"reply": reply}

    except Exception as e:
        logging.exception(e)
        return {"reply": "Server error"}