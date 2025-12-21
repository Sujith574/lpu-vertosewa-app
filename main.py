from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz

from google.cloud import firestore
from google import genai

# ------------------------------------------------------
# APP INIT
# ------------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------
# HEALTH CHECK (CLOUD RUN)
# ------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# ------------------------------------------------------
# GEMINI CONFIG
# ------------------------------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "models/gemini-2.5-flash"

# ------------------------------------------------------
# FIRESTORE (LAZY INIT)
# ------------------------------------------------------
def get_db():
    return firestore.Client()

# ------------------------------------------------------
# LOAD STATIC LPU KNOWLEDGE
# ------------------------------------------------------
def load_lpu_knowledge():
    try:
        with open("lpu_knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

STATIC_LPU = load_lpu_knowledge()

# ------------------------------------------------------
# SEARCH ADMIN CONTENT (CATEGORY + KEYWORDS)
# ------------------------------------------------------
def search_admin_content(question: str):
    db = get_db()
    q = question.lower()
    matches = []

    docs = (
        db.collection("lpu_content")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(50)
        .stream()
    )

    for doc in docs:
        d = doc.to_dict()

        text = (d.get("textContent") or "").lower()
        keywords = d.get("keywords") or []
        category = (d.get("category") or "").lower()

        if any(k in q for k in keywords) or category in q:
            matches.append(
                f"{d.get('title','')}:\n{d.get('textContent','')}"
            )

    return "\n\n".join(matches)

# ------------------------------------------------------
# GREETING
# ------------------------------------------------------
def handle_greeting(text: str):
    if text in ["hi", "hello", "hey", "hii", "hai", "namaste"]:
        return (
            "Hello! 👋\n\n"
            "You can ask about:\n"
            "• LPU exams, attendance, hostels, fees\n"
            "• RMS / UMS / registrations\n"
            "• DSW notices\n"
            "• UPSC, GK, people\n"
            "• Date & time"
        )
    return None

# ------------------------------------------------------
# GEMINI RESPONSE
# ------------------------------------------------------
def gemini_reply(question: str, context: str = ""):
    prompt = f"""
You are an educational assistant.

Rules:
- Reply only in English
- Be accurate, professional, and concise
- If LPU context is provided, strictly use it
- If no context is provided, answer from general knowledge
- Never say "context not provided"

LPU CONTEXT:
{context}

QUESTION:
{question}
"""
    try:
        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return res.text.strip()
    except Exception as e:
        logging.error(e)
        return "Please try again later."

# ------------------------------------------------------
# CORE MESSAGE LOGIC
# ------------------------------------------------------
def process_message(msg: str) -> str:
    text = msg.lower().strip()

    # 1️⃣ FIXED IDENTITIES (HIGHEST PRIORITY)
    if "sujith lavudu" in text:
        return (
            "Sujith Lavudu is a student innovator, software developer, and author. "
            "He is the co-creator of the LPU Vertosewa AI Assistant "
            "and co-author of the book 'Decode the Code'."
        )

    if "vennela barnana" in text:
        return (
            "Vennela Barnana is an author and researcher. "
            "She is the co-creator of the LPU Vertosewa AI Assistant "
            "and co-author of the book 'Decode the Code'."
        )

    if "rashmi mittal" in text:
        return (
            "Dr. Rashmi Mittal is the Pro-Chancellor of "
            "Lovely Professional University (LPU)."
        )

    # 2️⃣ GREETING
    greet = handle_greeting(text)
    if greet:
        return greet

    # 3️⃣ DATE / TIME
    if "time" in text or "date" in text:
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        return (
            f"📅 Date: {now.strftime('%d %B %Y')}\n"
            f"⏰ Time: {now.strftime('%I:%M %p')} (IST)"
        )

    # 4️⃣ BOT IDENTITY
    if any(k in text for k in [
        "who developed you",
        "who created you",
        "your developer",
        "your creator"
    ]):
        return (
            "I was developed by Sujith Lavudu and Vennela Barnana "
            "for Lovely Professional University (LPU)."
        )

    # 5️⃣ LPU-FIRST LOGIC
    LPU_TERMS = [
        "lpu", "lovely professional university",
        "ums", "rms", "dsw",
        "attendance", "hostel", "fees",
        "exam", "semester", "registration",
        "reappear", "mid term", "end term"
    ]

    if any(k in text for k in LPU_TERMS):
        admin_answer = search_admin_content(msg)

        if admin_answer.strip():
            return gemini_reply(msg, admin_answer)

        if STATIC_LPU.strip():
            return gemini_reply(msg, STATIC_LPU)

        return "No official LPU update is available for this query yet."

    # 6️⃣ GENERAL QUESTIONS → GEMINI
    return gemini_reply(msg)

# ------------------------------------------------------
# FLUTTER CHAT API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Please enter a valid question."}

    return {"reply": process_message(message)}