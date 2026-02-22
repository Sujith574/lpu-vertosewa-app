from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz
from google import genai

# ------------------------------------------------------
# APP INIT
# ------------------------------------------------------
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


def gemini_reply(question: str):
    prompt = f"""
You are an AI assistant.

Answer the question clearly and directly.
Do not explain reasoning.
Give only the final answer.

Question:
{question}

Final Answer:
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
        return "Sorry, I couldn’t process that right now."


# ------------------------------------------------------
# LOAD STATIC LPU KNOWLEDGE
# ------------------------------------------------------
def load_lpu_knowledge():
    try:
        with open("lpu_knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        logging.warning("lpu_knowledge.txt not found")
        return ""

STATIC_LPU = load_lpu_knowledge()


# ------------------------------------------------------
# LPU DETECTION
# ------------------------------------------------------
def is_lpu_related(question: str) -> bool:
    lpu_keywords = [
        "lpu", "lovely professional university",
        "ums", "rms", "dsw",
        "attendance", "hostel", "fees",
        "exam", "semester", "registration",
        "reappear", "mid term", "end term",
        "student organization", "soc",
        "mooc", "nptel", "swayam"
    ]

    text = question.lower()
    return any(k in text for k in lpu_keywords)


# ------------------------------------------------------
# STATIC SEARCH LOGIC
# ------------------------------------------------------
def search_lpu_knowledge(question: str, knowledge: str) -> str:
    if not knowledge.strip():
        return ""

    q_words = [w for w in question.lower().split() if len(w) > 3]
    chunks = knowledge.split("\n\n")

    best_match = ""
    best_score = 0

    for chunk in chunks:
        chunk_l = chunk.lower()
        score = sum(1 for w in q_words if w in chunk_l)

        if score > best_score:
            best_score = score
            best_match = chunk

    if best_score >= 3:
        return best_match.strip()

    return ""


# ------------------------------------------------------
# GREETING
# ------------------------------------------------------
def handle_greeting(text: str):
    if any(text.startswith(g) for g in ["hi", "hello", "hey", "hai", "namaste"]):
        return (
            "Hello 👋 I’m LPU VertoSewa.\n\n"
            "Ask me anything related to academics, exams, UMS, hostels, "
            "fees, or university information."
        )
    return None


# ------------------------------------------------------
# TIME & DATE
# ------------------------------------------------------
def handle_time_date(text: str):
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if "time" in text and "date" not in text:
        return f"Time: {now.strftime('%I:%M %p')} (IST)"

    if "date" in text:
        return f"Date: {now.strftime('%d %B %Y')}"

    return None


# ------------------------------------------------------
# CORE MESSAGE PROCESSOR
# ------------------------------------------------------
def process_message(msg: str) -> str:
    text = msg.lower().strip()

    # 1. Greeting
    greeting = handle_greeting(text)
    if greeting:
        return greeting

    # 2. Time / Date
    time_reply = handle_time_date(text)
    if time_reply:
        return time_reply

    # 3. LPU Flow
    if is_lpu_related(msg):

        # Step 1: Try static data
        static_answer = search_lpu_knowledge(msg, STATIC_LPU)

        if static_answer:
            return static_answer

        # Step 2: Fallback to Gemini
        return gemini_reply(msg)

    # 4. Non-LPU → Direct Gemini
    return gemini_reply(msg)


# ------------------------------------------------------
# CHAT API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Please enter a valid question."}

    return {"reply": process_message(message)}