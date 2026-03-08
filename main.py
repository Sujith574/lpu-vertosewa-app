from fastapi import FastAPI, Request
import os
import logging
from google import genai

app = FastAPI()
logging.basicConfig(level=logging.INFO)

MODEL = "models/gemini-2.5-flash"


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("API key missing")

    return genai.Client(api_key=api_key)


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/chat")
async def chat(request: Request):

    data = await request.json()
    message = data.get("message")

    if not message:
        return {"reply": "Send a message."}

    try:

        client = get_client()

        response = client.models.generate_content(
            model=MODEL,
            contents=message
        )

        reply = response.text

        return {"reply": reply}

    except Exception as e:
        logging.exception(e)
        return {"reply": "Server error"}