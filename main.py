from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz
from google import genai

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# ------------------------------------------------------
# GEMINI CONFIG
# ------------------------------------------------------
GEMINI_MODEL = "models/gemini-2.5-flash"

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

# ------------------------------------------------------
# LOAD LPU KNOWLEDGE
# ------------------------------------------------------
def load_lpu_knowledge():
    try:
        with open("lpu_knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

STATIC_LPU = load_lpu_knowledge()

# ------------------------------------------------------
# GEMINI RESPONSE
# ------------------------------------------------------
def gemini_reply(question: str):

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    prompt = f"""
You are **LPU VertoSewa**, an AI assistant for Lovely Professional University.

Your job is to help students with university information.

Use the knowledge below if relevant.

-----------------------------------
LPU KNOWLEDGE
{STATIC_LPU}
-----------------------------------

Current Time: {now.strftime('%I:%M %p')}
Current Date: {now.strftime('%d %B %Y')}

Rules:
- Answer clearly
- Be short and direct
- If the question is not about LPU, still answer normally

User Question:
{question}

Answer:
"""

    try:
        client = get_gemini_client()

        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return res.text.strip()

    except Exception:
        logging.exception("Gemini error")
        return "Sorry, I couldn’t process that."

# ------------------------------------------------------
# CHAT API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):

    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Please enter a question."}

    answer = gemini_reply(message)

    return {"reply": answer}